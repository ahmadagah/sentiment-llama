Notes on the code.
1. You want to look at LLMs-25-QLoRA.py.
2. It asks you if you want to train the model, then decides which thing it's going to do. 
2. The directory where the LoRA is stored is hard-coded -- in two places. So if you want to run it, you'll need to mess with that. Ditto the model it's loading from HF.
3. Hopefully it's not too messy. I've made some changes that I probably haven't properly cleaned up from -- like getting the HF token from the .env file, which I learned from Ahmad.
4. Jupyter notebooks are also a great idea, but I haven't tried to integrate them since finding out about them.


These are just my notes for the homework report, but you can see accuracies here.


    base_prompt = 'Evaluate for sentiment (neutral, positive, negative): '
zero-shot:
              precision    recall  f1-score   support

    negative       0.66      0.88      0.76       273
     neutral       0.58      0.47      0.52       255
    positive       0.79      0.67      0.72       274

    accuracy                           0.68       802
   macro avg       0.68      0.67      0.67       802
weighted avg       0.68      0.68      0.67       802
Invalid: 68

one-shot (neutral is first in the prompt)
              precision    recall  f1-score   support
    negative       0.64      0.87      0.74       284
     neutral       0.84      0.10      0.17       281
    positive       0.60      0.90      0.72       288

    accuracy                           0.62       853
   macro avg       0.69      0.62      0.54       853
weighted avg       0.69      0.62      0.54       853

three-shot:
              precision    recall  f1-score   support
    negative       0.71      0.78      0.74       289
     neutral       0.54      0.47      0.50       289
    positive       0.70      0.73      0.72       290

    accuracy                           0.66       868
   macro avg       0.65      0.66      0.65       868
weighted avg       0.65      0.66      0.65       868
Invalid: 2

negative 319
neutral 249
positive 300




Parameters:
Learning rate: 2e-4
Batch size: 4
Epochs: 3

Example:
Git 'em girls #BarackObama #blm #blacklivesmatter #mylifematters #therealskinnysuge #thepeopleschamp #skinnyup #pmd
L: Neutral
1: Negative
3: Positive
F: Neutral
Z: Positive

Alfred Essa, VP Analytics & R&D @user talks about "Deep Learning Primer for Business Leaders" @ #PAChicago N\u2026
label: neutral
One: positive
Three: neutral
Fine: neutral
Zero: invalid


Kelly Anne Conway is here reporting the wall is a glorious 10 meters high, while Ben Carson counters that 'meters' arent even a real thing
Label: positive
One: negative
Three: neutral
Fine: neutral
Zero: negative

 I'm crying over Richard and Leonard Cohen 😭😭😭 #GilmoreGirlsRevival
 Label: neutral
 One:   positive
 Three: positive
 Fine:  negative
 Zero:  negative