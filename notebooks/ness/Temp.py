# Ness Blackbird homework 2.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from torch.nn.utils.rnn import pad_sequence
ds = load_dataset(
    "cardiffnlp/tweet_sentiment_multilingual",
    "english",
    trust_remote_code=True
)

model_name = 'meta-llama/Llama-3.2-1B-Instruct'
token = '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'
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
batch_size = 8
# Set up a tuple for the responses. These correspond to the labels in the dataset. I added invalid.
sentiments = {'negative': 0, 'neutral': 1, 'positive': 2, 'invalid': 3}
counts = {s: 0 for s in sentiments}
accurate = 0
n = 0

def extract_sentiment(output, label, prompt):
    global accurate
    print('----------New prompt-----------')
    print('Prompt: ', prompt)
    print('Output: ', output)
    # Find any output matching the qualifiers.
    matches = re.findall(r"(positive|negative|neutral)", output.lower(), re.IGNORECASE)
    # Remove duplicates.
    matches = list(set(matches))
    if len(matches) == 1:
        # We have a valid response.
        counts[matches[0]] += 1
        print('Response: ', matches[0], label)
        if sentiments[matches[0]] == label:
            accurate += 1
    else:
        # No valid response.
        counts['invalid'] += 1
        print('Response invalid')


for i in range(0, len(texts), batch_size):
    batch_texts = texts[i : i + batch_size]
    batch_labels = labels[i: i + batch_size]

    # prompts = [[
    #     {'role': 'user', 'content': 'Sentiment: "This movie was amazing!" Negative, Neutral, or Positive?'},
    #     {'role': 'assistant', 'content': 'Positive'},
    #     {'role': 'user', 'content': 'Sentiment: "It was okay, nothing special." Negative, Neutral, or Positive?'},
    #     {'role': 'assistant', 'content': 'Neutral'},
    #     {'role': 'user', 'content': f'Sentiment: "{text}" Negative, Neutral, or Positive?'}
    # ] for text in batch_texts]

    prompts = [[
        {'role': 'user', 'content': f'Is this text negative, neutral, or positive? Answer with just one word. "{text}"'}
    ] for text in batch_texts]


    input_ids = []
    for prompt in prompts:
        tokens = (tokenizer.apply_chat_template(
            prompt,
            return_tensors = 'pt',
            add_generation_prompt = True,
        ))
        input_ids.append(tokens.squeeze(0))

    input_ids = pad_sequence(
        input_ids,
        batch_first   = True,
        padding_value = tokenizer.pad_token_id
    ).to(model.device)

    # Create attention mask (1 for real tokens, 0 for padding).
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

    outputs = model.generate(
        input_ids,
        attention_mask = attention_mask,
        max_new_tokens = 100,
        temperature    = 0.3,
        do_sample      = True,
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
    n += 1


for sentiment in sentiments:
    print(sentiment, counts[sentiment])
percent = accurate / len(texts) * 100
print('Accurate: ', accurate, 'of', len(texts), f", {percent:.0f}%")
