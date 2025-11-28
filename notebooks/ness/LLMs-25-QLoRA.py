# Ness Blackbird homework Group Project: QLoRA.
import torch
import re
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
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
MODEL_ID: str = _get_env("MODEL_ID", default = "meta-llama/Meta-Llama-3-8B-Instruct")
# Get the dataset.
training_dataset = load_dataset(
    "cardiffnlp/tweet_sentiment_multilingual",
    "english",
    trust_remote_code=True
)

# Used to build all the prompts.
base_prompt = 'Evaluate for sentiment (neutral, positive, negative): '

# These correspond to the labels in the dataset. I added invalid.
sentiments = {'negative': 0, 'neutral': 1, 'positive': 2, 'invalid': 3}
sentiments_reversed = ('negative', 'neutral', 'positive', 'invalid')

training = int(input("Train the model? Enter 0 or 1: "))

# QLoRA.
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,  # Basic QLoRA.
    bnb_4bit_use_double_quant = True,  # Also pretty basic, but the double quant thing is interesting.
    bnb_4bit_quant_type       = "nf4", # The 4-bit number format.
    bnb_4bit_compute_dtype    = torch.bfloat16
)

checkpoint = "./qlora-8B-2a"

# ------------------------- Train model -------------------------
if training:
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        token               = HF_token,
        device_map          = "auto"
    ).to('cuda')

    model = prepare_model_for_kbit_training(model)

    # Add LoRA to the model. First make a LoRA.
    lora_config = LoraConfig(
        r              = 4,
        lora_alpha     = 8,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.15,
        bias           = "none",
        task_type      = "CAUSAL_LM"
    )

    # Wrap the model in it.
    model = get_peft_model(model, lora_config)
    print ('Model device: ', model.device)

    # Build a training configuration.
    training_args = TrainingArguments(
        output_dir                  = checkpoint + "_out",
        num_train_epochs            = 1,
        per_device_train_batch_size = 1,        # QLoRA uses a lot of memory.
        per_device_eval_batch_size  = 1,
        learning_rate               = 7e-5,
        gradient_accumulation_steps = 4,
        eval_accumulation_steps     = 4,
        gradient_checkpointing      = True,
        optim                       = "paged_adamw_8bit",
        fp16                        = False,
        bf16                        = True,

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HF_token)
    tokenizer.pad_token = tokenizer.eos_token

    # ----------------------- training_format ------------------------
    def training_format(data):
        # Data contains a dict, which contains lists: {'text': [list of tweets], 'label': [list of labels]}.

        # Convert the labels to text.
        labels = [sentiments_reversed[label] for label in data['label']]

        prompts = [[
            {'role': 'user', 'content': base_prompt + '"' + text + '"'},
            {'role': 'assistant', 'content': label}] for text, label in zip(data['text'], labels)
        ]

        # Format them with the chat template.
        formatted_prompts = [tokenizer.apply_chat_template(prompt, tokenize = False) for prompt in prompts]

        # Tokenize them.
        encoded = tokenizer(formatted_prompts, return_attention_mask = True)

        # Now to make the correct mask for the labels, we need to seriously mess around.
        # 1. Rerun the tokenization without the assistant row.
        user_prompts = [[
            {'role': 'user', 'content': base_prompt + '"' + text + '"'}] for text in data['text']
        ]

        # 2. Format them with the chat template, too. This is just to get the number of user tokens.
        formatted_user_prompts = [tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in user_prompts]
        user_prompt_counts = [len(ids) for ids in tokenizer(formatted_user_prompts)['input_ids']]

        # 3. Mask the user tokens in the "real" tokens to make the label tokens.
        # Slice out the initial user_prompt_count input_ids (tokens) and replace them with -100.
        # This is the way the model expects to see the labels: With everything other than the label masked out.
        # Now run through the pieces and zip them together.
        labels_output = []
        for input_ids, count in zip(encoded['input_ids'], user_prompt_counts):
            # This makes a shallow copy. But input_ids is just a list of integers, so it's fine.
            labels = input_ids.copy()
            # Mask the prompt in the copy.
            labels[:count] = [-100] * count
            labels_output.append(labels)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels_output
        }


    def compute_metrics_callback(eval_pred):
        predictions, labels = eval_pred
        # Move to CPU and convert to numpy if needed
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        predictions = predictions.argmax(axis =- 1)
        return {"accuracy": accuracy_score(labels, predictions)}


    # Tokenize it using the training_format function.
    tokenized_data = training_dataset.map(training_format, batched = True, remove_columns=['text', 'label'])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer          = tokenizer,
        model              = model,
        padding            = True,
        label_pad_token_id = -100
    )

    # And initialize the trainer.
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = tokenized_data["train"],
        eval_dataset    = tokenized_data["validation"],
        data_collator   = data_collator,
        compute_metrics = compute_metrics_callback
    )

    print(type(tokenized_data['train'][0]['labels']))
    print(tokenized_data['train'][0]['labels'][:10])
    trainer.train()

    # Save the weights.
    trainer.save_model(checkpoint)
else:
    # ------------------------- Use saved model ----------------------------
    # The model is already trained. Load it.
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config = bnb_config,
        device_map          = "auto"
    ).to('cuda')

    # Separately load the LoRA adapter.
    model = PeftModel.from_pretrained(base_model, checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token = HF_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    test_data = training_dataset['train']
    texts = test_data['text']
    labels = test_data['label']

    predictions = []
    batch_size = 8
    counts = {s: 0 for s in sentiments}
    accurate = 0

    # ------------------------------- extract_sentiment -----------------------------
    def extract_sentiment(output, label, prompt):
        global accurate
        # Find any output matching the qualifiers.
        matches = re.findall(r"(positive|negative|neutral)", output.lower(), re.IGNORECASE)
        # Remove duplicates.
        matches = list(set(matches))
        if len(matches) == 1:
            # We have a valid response.
            counts[matches[0]] += 1
            if sentiments[matches[0]] == label:
                accurate += 1
            model = matches[0]
        else:
            # No valid response.
            counts['invalid'] += 1
            model = 3
        return {'output': output, 'label': label, 'model': model}


    # Prepare and generate predictions.
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i: i + batch_size]

        base_prompt = 'Evaluate for sentiment (neutral, positive, negative): '
        prompts = [[
            {'role': 'user', 'content': base_prompt + '"' + text + '"'}
        ] for text in batch_texts]

        encoded = tokenizer(
            [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
             for prompt in prompts],
            return_tensors        = 'pt',
            padding               = True,
            return_attention_mask = True
        ).to(model.device)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        outputs = model.generate(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = 20,
            do_sample      = False,
            pad_token_id   = tokenizer.pad_token_id
        )

        # Slice off the input tokens. That says, "for all input rows, remove everything before
        # the length of the input matrix". We do this before decoding because skip_special_tokens
        # will cut off the padding and make the rows different lengths -- they're padded going into the model.
        new_tokens = outputs[:, input_ids.shape[1]:]
        outputs_decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        batch_predictions = [extract_sentiment(output, label, prompt) for output, label, prompt in
                             zip(outputs_decoded, batch_labels, prompts)]
        predictions.extend(batch_predictions)


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
        data[data_name] = [pred['output'] for pred in predictions]

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


    correct_predictions = [pred for pred in predictions if pred['output'].lower() == sentiments_reversed[pred['label']]]
    for sentiment in sentiments:
        print(sentiment, counts[sentiment])

    # Calculate F1 score using sklearn.
    labels = [pred['label'] for pred in predictions]
    predicted_labels = [sentiments[pred['output']] for pred in predictions]

    percent = len(correct_predictions) / len(texts)

    print('Accurate: ', len(correct_predictions), 'of', len(texts), f"{percent:.2%}")

    report = classification_report(labels, predicted_labels, target_names=['negative', 'neutral', 'positive'])
    print (report)