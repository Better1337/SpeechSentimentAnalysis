import pandas as pd
import re

columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=columns)


data = data[['target', 'text']]
data = data[data['target'] != 2]
data['target'] = data['target'].map({0: 0, 4: 1})

def clean_text(text):
    text = re.sub(r"http\S+", "", text)           
    text = re.sub(r"@\w+", "", text)             
    text = re.sub(r"#\w+", "", text)            
    text = re.sub(r"[^\w\s]", "", text)         
    text = re.sub(r"\d+", "", text)    
    text = text.lower() 
    text = text.strip() 
    return text

data['text'] = data['text'].apply(clean_text)

data.to_csv('cleaned_dataset.csv', index=False)



