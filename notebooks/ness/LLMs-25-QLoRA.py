# Ness Blackbird homework Group Project: QLoRA.
import numpy as np
import torch
import re
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorWithPadding, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import classification_report, accuracy_score
import os
import pickle
import pandas as pd
from dotenv import load_dotenv

# Get the HF token.
load_dotenv()

def _get_env(name: str, *, default=None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or v == ""):
        raise EnvironmentError(f"Missing required env var: {name}")
    return v

# Inconsistent capitalization on HF_token is apparently normal?
HF_token = _get_env("HF_TOKEN", required = True)
MODEL_ID: str = _get_env("MODEL_ID", default = "meta-llama/Meta-Llama-3-8B")
# Get the dataset.
training_dataset = load_dataset(
    "cardiffnlp/tweet_sentiment_multilingual",
    "english",
    trust_remote_code=True
)

# Used to build all the prompts.
base_prompt = 'Evaluate for sentiment (negative, neutral, positive): '

# These correspond to the labels in the dataset. I added invalid.
sentiments = {'negative': 0, 'neutral': 1, 'positive': 2, 'invalid': 3}
sentiments_reversed = ('negative', 'neutral', 'positive', 'invalid')

# QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,  # Basic QLoRA.
    bnb_4bit_use_double_quant = True,  # Also pretty basic, but the double quant thing is interesting.
    bnb_4bit_quant_type       = "nf4", # The 4-bit number format.
    bnb_4bit_compute_dtype    = torch.bfloat16
)

checkpoint = "./qlora-8B-2a"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

training = int(input("Train the model? Enter 0 or 1: "))
if training:
    # ------------------------- Train model -------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        token               = HF_token,
        num_labels          = 3,
        device_map          = "auto"
    ).to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Required for gradient checkpointing
    model.config.pretraining_tp = 1  # Recommended for reproducibility

    model = prepare_model_for_kbit_training(model)

    # Add LoRA to the model. First make a LoRA.
    lora_config = LoraConfig(
        r               = 8,
        lora_alpha      = 16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout    = 0.1,
        bias            = "none",
        task_type       = TaskType.SEQ_CLS,
        modules_to_save = ["score"]
    )

    # Wrap the model in it.
    model = get_peft_model(model, lora_config)
    print ('Model device: ', model.device)

    # Build a training configuration.
    training_args = TrainingArguments(
        output_dir                  = checkpoint + "_out",
        num_train_epochs            = 2,
        per_device_train_batch_size = 1,        # QLoRA uses a lot of memory.
        per_device_eval_batch_size  = 1,
        learning_rate               = 5e-5,
        gradient_accumulation_steps = 4,
        eval_accumulation_steps     = 4,
        gradient_checkpointing      = True,
        optim                       = "paged_adamw_8bit",
        fp16                        = False,
        bf16                        = True,
        metric_for_best_model       = 'eval_loss',

        # Logging configuration
        logging_dir                 = checkpoint + "_out/logs",
        logging_strategy            = "steps",
        logging_steps               = 50,

        # Evaluation configuration
        eval_strategy               = "steps",
        eval_steps                  = 200,

        save_strategy               = "steps",
        save_steps                  = 200,
        load_best_model_at_end      = True
    )

    def compute_metrics_callback(eval_pred):
        predictions, labels = eval_pred

        # Debug: check what we actually received
        print(f"Type of predictions: {type(predictions)}")
        print(
            f"Predictions shape before processing: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")

        # Handle tuple case - take only the logits
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Move to CPU and convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        predictions = predictions.argmax(axis=-1)

        print(f"Predictions shape after argmax: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")

        return {"accuracy": accuracy_score(labels, predictions)}

    # ----------------------- Tokenize ------------------------
    def tokenize_function(data):
        full_text = [base_prompt + t for t in data["text"]]
        tokens = tokenizer(full_text, truncation=True)
        # Include the label from the dataset. It gets renamed.
        tokens["labels"] = data["label"]
        return tokens

    # Tokenize it using the training_format function.
    tokenized_data = training_dataset.map(tokenize_function, batched = True)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding (tokenizer = tokenizer)

    # And initialize the trainer.
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized_data["train"],
        eval_dataset    = tokenized_data["validation"],
        data_collator   = data_collator,
    )

    trainer.train()

    # Save the weights.
    trainer.save_model(checkpoint)
else:
    # ------------------------- Use saved model ----------------------------
    # The model is already trained. Load it.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto",
        num_labels = 3,
    ).to('cuda')

    # Separately load the LoRA adapter.
    model = PeftModel.from_pretrained(base_model, checkpoint)

    test_data = training_dataset['test']
    texts = test_data['text']
    labels = test_data['label']

    predictions = []
    batch_size = 1
    accurate = 0

    # Prepare and generate predictions.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i: i + batch_size]

        prompts = [base_prompt + '"' + text + '"' for text in batch_texts]

        encoded = tokenizer(
            prompts,
            return_tensors        = 'pt',
            padding               = True,
            return_attention_mask = True
        ).to(model.device)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            pad_token_id   = tokenizer.pad_token_id
        )

        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)
        predictions.append({'output': pred.tolist()[0], 'label': batch_labels[0], 'text': batch_texts[0]})


    # Save the data for comparison.
    data_name = input('What should I call this run? Enter to not save. ')
    if data_name:
        # Load previously saved data if it exists.
        if os.path.exists('sentiment-data.pkl'):
            with open('sentiment-data.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            # First time: Create the structure.
            data = dict()
            data['texts'] = texts
            data['labels'] = [sentiments_reversed[lab] for lab in labels]

        # Add in the current version.
        data[data_name] = predictions

        # Save it.
        with open('sentiment-data.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open('sentiment-data.pkl', 'rb') as f:
            data = pickle.load(f)

    # Now that it's saved, we build a pandas DataFrame table to do comparisons.
    comparison = pd.DataFrame(data)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    print(comparison.head(50))

    # Print a classification report.
    labels = [pred['label'] for pred in predictions]
    predicted_labels = [pred['output'] for pred in predictions]

    report = classification_report(labels, predicted_labels, target_names=['negative', 'neutral', 'positive'])
    print (report)