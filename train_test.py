import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer

data=pd.read_csv('cleaned_dataset.csv')

X = data['text'].values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1410)
