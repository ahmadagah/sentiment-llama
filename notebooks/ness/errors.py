# Ness Blackbird homework 2.
import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


with open('sentiment-data.pkl', 'rb') as f:
    data = pickle.load(f)

style = 'Fine-tuning'
disp = ConfusionMatrixDisplay.from_predictions(data['labels'], data[style])
disp.plot()
plt.savefig(style + '.png')  # save instead of show

#errors = [(label, pred) for label, pred in zip(data['labels'], data['Fine-tuning']) if label != pred]
cm = confusion_matrix(data['labels'], data['Fine-tuning'])
#print(cm)