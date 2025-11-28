# Ness Blackbird homework 2, base model.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report
import os
import pickle
import pandas as pd

ds = load_dataset(
    "cardiffnlp/tweet_sentiment_multilingual",
    "english",
    trust_remote_code=True
)

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token = token)
tokenizer.padding_side = 'left'
# It needs to be told to use its end-of-sequence token for padding.
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, token = token, device_map = "auto")
print ('Model device: ', model.device)

# Get the test data.
test_data = ds['test']
texts = test_data['text']
labels = test_data['label']

predictions = []
batch_size = 16
# These correspond to the labels in the dataset. I added invalid.
sentiments = {'negative': 0, 'neutral': 1, 'positive': 2, 'invalid': 3}
sentiments_reversed = ('negative', 'neutral', 'positive', 'invalid')

counts = {s: 0 for s in sentiments}
accurate = 0

def extract_sentiment(output, label, prompt):
    # Figure out what the model is trying to tell us. It unfortunately doesn't usually output
    # just a one-word answer, but most of the time, the answer does include an answer, and just one.
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
        output = matches[0]
    else:
        # No valid response.
        counts['invalid'] += 1
        output = 'invalid'
    return {'output': output, 'label': label}


for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    batch_labels = labels[i: i + batch_size]

    # This is the 3-shot version. I'm not using any of the included neutral data points, because
    # they tend to confuse it. Better results this way.
    base_prompt = 'Evaluate for sentiment (neutral, positive, negative): '
    prompts = [[
        # {'role': 'user', 'content': base_prompt + '"Trying to have a conversation with my dad about vegetarianism is the most pointless infuriating thing ever #caveman "'},
        # {'role': 'assistant', 'content': 'negative'},
        # {'role': 'user', 'content': base_prompt + '"Um, I dunno, it was fine, I guess. "'},
        # {'role': 'assistant', 'content': 'neutral'},
        # {'role': 'user', 'content': base_prompt + '"@user You are a stand up guy and a Gentleman Vice President Pence "'},
        # {'role': 'assistant', 'content': 'positive'},
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
    # for index, row in comparison.iterrows():
    #     t = [f"{col_name}: {row[col_name]}" for col_name in row]
    #     t = ' '.join(t)
    #     for col_name in df.columns:
    #         print(f"{col_name}: {row[col_name]}")

for sentiment in sentiments:
    print(sentiment, counts[sentiment])

labels = [pred['label'] for pred in predictions if pred['output'] != 'invalid']
predicted_labels = [sentiments[pred['output']] for pred in predictions if pred['output'] != 'invalid']

report = classification_report(labels, predicted_labels,
   target_names=['negative', 'neutral', 'positive'])

print(report)