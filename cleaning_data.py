import os

import pandas as pd
import re
from langdetect import detect
import numpy as np
import string
from unicodedata import category
import sys
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz


def det_lang(df):
    '''
    :param df: dataframe
    :return: dataframe with added language column
    '''
    def detect_language(text):
        try:
            lan = detect(text)
        except:
            lan = 'unknown'
        return lan

    df['language'] = df['text'].apply(detect_language)
    return df


def delete_stopwords(text):
    '''
    :param text: string
    :return: string with deleted stopwords
    '''
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def clean_text(text, punctuation_chars):
    '''
    :param text: string
    :param punctuation_chars: string with all punctuaction characters to remove
    :return: processed text
    '''
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', ''.join(punctuation_chars)))
    # remove digits
    text = text.translate(str.maketrans('', '', string.digits))
    # remove all single characters
    pattern = r'(^| ).( |$)'
    text = re.sub(pattern, ' ', text)
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    # remove stopwords
    text = delete_stopwords(text)
    # stemming
    text = stemming(text)
    return text


def clean_df(df):
    '''
    :param df: dataframe to process
    :return: preprocessed dataframe
    '''

    df = df[['title', 'text', 'Ground Label']]
    df['full_text'] = df['title'] + ' ' + df['text']
    df = df.drop(['title', 'text'], axis=1)

    df.dropna(subset=['full_text'], inplace=True)
    df = df.fillna('')

    punctuation_chars = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P")]

    df['full_text'] = df['full_text'].map(lambda x: clean_text(x, punctuation_chars))
    df = df[df['full_text'].str.len() >= 30]
    return df


def stemming(text):
    '''
    :param text: text to be stemmed
    :return: stemmed text
    '''
    words = word_tokenize(text)
    porter = PorterStemmer()
    stem_words = [porter.stem(word) for word in words]
    return ' '.join(stem_words)


def delete_not_english(df):
    '''
    :param df: dataframe
    :return: dataframe with removed rows containing not english texts
    '''
    df = det_lang(df)
    df = df[df['language'] == 'en']
    df = df.rename(columns={df.columns[0]: 'id'})
    df = df.drop(['id', 'language'], axis=1)
    return df


def split_data(df):
    x = df['full_text']
    y = df['Ground Label']

    # splitting data into train and test sets (and then splitting train test into train and test for us)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=123,
                                                        test_size=0.3,
                                                        shuffle=True)
    x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train,
                                                                                random_state=123,
                                                                                test_size=0.3,
                                                                                shuffle=True)
    return x_train_train, x_train_test, x_test, y_train_train, y_train_test, y_test


def tfidf(x_train_train, x_train_test, x_test):
    tfidfvectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train_train = tfidfvectorizer.fit_transform(x_train_train)
    tfidf_train_test = tfidfvectorizer.transform(x_train_test)
    tfidf_test = tfidfvectorizer.transform(x_test)
    return tfidf_train_train, tfidf_train_test, tfidf_test


def create_directories():
    try:
        os.mkdir('validation_data')
        print('Directory validation_data created')
    except FileExistsError:
        print('Directory validation_data already exists')

    try:
        os.mkdir('train_data')
        print('Directory train_data created')
    except FileExistsError:
        print('Directory train_data already exists')


def save_to_files(x_train_train, x_train_test, x_test, y_train_train, y_train_test, y_test):

    y_test.to_csv('validation_data/y_test.csv', encoding='utf-8')
    save_npz('validation_data/x_test.npz', x_test)
    print('validation data saved')
    y_train_train.to_csv('train_data/y_train_train.csv', encoding='utf-8')
    save_npz('train_data/x_train_train.npz', x_train_train)
    y_train_test.to_csv('train_data/y_train_test.csv', encoding='utf-8')
    save_npz('train_data/x_train_test.npz', x_train_test)
    print('test data saved')
    print('success! c:')


def main():
    # language detection is slow so i saved it for tests
    #df = pd.read_csv('original_data.csv')
    #df = delete_not_english(df)
    #df.to_csv('en_data.csv')

    ### test ###
    #df = pd.read_csv('en_data.csv')
    #df = clean_df(df)
    #df.to_csv('clean_data.csv')
    # df = pd.read_csv('clean_data.csv')
    #
    # print(df['full_text'][0])
    #
    # x_train_train, x_train_test, x_test, y_train_train, y_train_test, y_test = split_data(df)
    # x_train_train, x_train_test, x_test = tfidf(x_train_train, x_train_test, x_test)
    # save_to_files(x_train_train, x_train_test, x_test, y_train_train, y_train_test, y_test)

    tfidf_1 = load_npz('train_data/x_train_train.npz')
    print(tfidf_1)


if __name__ == "__main__":
    main()
