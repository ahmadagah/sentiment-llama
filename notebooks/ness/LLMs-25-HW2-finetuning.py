# Ness Blackbird homework 2.
import torch
import re
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from sklearn.metrics import classification_report
import os
import pickle
import pandas as pd

# Get the dataset.
HF_token = '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'
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

# ------------------------- Train model -------------------------
if training:
    # Train the model, unless it's already trained and saved.
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token = HF_token,
    ).to('cuda')

    # Add LoRA to the model. First make a LoRA.
    lora_config = LoraConfig(
        r              = 32,
        lora_alpha     = 16,
        target_modules = ["q_proj", "k_proj"],  # We're adapting Q and K, at least to start with.
        lora_dropout   = 0.1,
        bias           = "none",
        task_type      = "CAUSAL_LM"
    )

    # Wrap the model in it.
    model = get_peft_model(model, lora_config)
    print ('Model device: ', model.device)

    # Build a training configuration.
    training_args = TrainingArguments(
        output_dir                  = "./llama-sentiment-lora",
        num_train_epochs            = 5,
        per_device_train_batch_size = 16,
        learning_rate               = 2e-4,

        # Logging configuration
        logging_dir                 = "./logs",
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
        model         = model,
        args          = training_args,
        train_dataset = tokenized_data["train"],
        eval_dataset  = tokenized_data["validation"],
        data_collator = data_collator
    )

    print(type(tokenized_data['train'][0]['labels']))
    print(tokenized_data['train'][0]['labels'][:10])
    trainer.train()

    # Save the weights.
    trainer.save_model("./lora-sentiment-model-5ep")
else:
    # ------------------------- Use saved model ----------------------------
    # The model is already trained. Load it.
    model_name = "./lora-sentiment-model-5ep"
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16
    ).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_name, token = HF_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    test_data = training_dataset['test']
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
            return_tensors='pt',
            padding=True,
            return_attention_mask=True
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