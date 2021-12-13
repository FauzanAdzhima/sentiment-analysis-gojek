from flask import Flask, flash, request, url_for, redirect, render_template, jsonify
import os
from os.path import join, dirname, realpath
import nltk
nltk.download('stopwords')
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Initiate Flask app
app = Flask(__name__)

# Enable debug mode
app.config['DEBUG'] = True

# Upload folder
UPLOAD_FOLDER = 'Temp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
with open('model_sentimen_gojek.pkl', 'rb') as model_init:
    model = pickle.load(model_init)

# Load vectorizer
with open('count_vectorized.pkl', 'rb') as count_vec_init:
    count_vec = pickle.load(count_vec_init)

# Load tfidf transformer
with open('tfidf_transform.pkl', 'rb') as tfidf_init:
    tfidf = pickle.load(tfidf_init)

# Root URL
@app.route('/')
def index():
    return render_template('index.html')

# Get uploaded files
@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files.get('input-dataset')
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        dataset = parseCSV(file_path)
        # transform_countvec = count_vec.transform([dataset.toarray()])
        # transform_tfidf = tfidf.transform(transform_countvec)
        # predict_result = model.predict(transform_tfidf)

        # bow_transformer = CountVectorizer()
        # count_vec = bow_transformer.fit_transform(dataset)
        a = ''

        if dataset.shape[0] < 500:
            a = 500 - dataset.shape[0]
            for x in range(0, a):
                df_count = dataset.shape[0]
                temp_series = pd.Series(dataset.sample(n=1, random_state=1))
                dataset = dataset.append(temp_series, ignore_index=True)
        
        if dataset.shape[0] > 500:
            # dataset = dataset.sample(n=500, random_state=1)
            dataset = dataset[:500]

        text = dataset.map(' '.join)
        count_vec = CountVectorizer()
        cv = count_vec.fit(text)
        cv_transform = cv.transform(text).toarray()        

        tf_transform = TfidfTransformer(use_idf=False).fit(cv_transform)
        tfidf = tf_transform.transform(cv_transform)
        


        # not_existing_cols = [c for c in dataset.tolist() if c not in tfidf]
        # tfidf = tfidf.reindex(tfidf.tolist() + not_existing_cols, axis=1)
        # tfidf.fillna(0, inplace=True)
        # tfidf = tfidf[dataset.tolist()]
        # tfidf = tfidf.fillna(0, inplace=True)

        # predict_result = model.predict(tfidf)

        return render_template('index.html', pred=model.predict(tfidf))
    else:
        return render_template('index.html', pred="Dataset tidak dapat dimuat")

# Read CSV file
def parseCSV(filepath):    
    df = pd.read_csv(filepath)
    df = df.rename(columns={'description': 'Tweet'})
    df = df[['Tweet', 'tweet_type']]
    # df = df[['Tweet']]
    df = df[df['tweet_type'] == 'original']
    new_df = df['Tweet']    
    return preprocessing(new_df)

# Text Preprocessing
def preprocessing(dataset):
    def clean_text(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # hilangkan @mention
        text = re.sub(r'#', '', text) # hilangkan simbol hashtag ('#')
        text = re.sub(r'RT[\s]+', '', text) # hilangkan RT (retweet)
        text = re.sub(r'https?:\/\/\S+', '', text) # hilangkan link
        text = re.sub(r'[\([{})\]]', '', text) # menghilangkan tanda kurung
        text = re.sub(r'[,"-.?:_/;\+=!“”]', '', text) # menghilangkan tanda baca
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'amp', '', text)

        # menghilangkan unicode escape character
        text = text.encode('ascii', 'ignore')
        text = text.decode()
        return text

    def tokenize(text):
        text = re.split('\W+', text)
        return text

    def stopword_remove(text):
        stopword = nltk.corpus.stopwords.words('indonesian')
        text = [word for word in text if word not in stopword]
        return text

    # dataset['clean_tweet'] = dataset['tweet'].apply(clean_text)
    # dataset['token_tweet'] = dataset['clean_tweet'].apply(tokenize)
    # dataset['sw_removed'] = dataset['token_tweet'].apply(stopword_remove)

    dataset = dataset.apply(clean_text)
    dataset = dataset.apply(tokenize)
    dataset = dataset.apply(stopword_remove)

    return dataset
    
if __name__ == '__main__':
    app.run()
