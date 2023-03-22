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
    df.dropna(subset=['title', 'text'], inplace=True)
    df = df.fillna('')

    punctuation_chars = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P")]

    df['text'] = df['text'].map(lambda x: clean_text(x, punctuation_chars))
    df['title'] = df['title'].map(lambda x: clean_text(x, punctuation_chars))
    df = df[df['text'].str.len() >= 30]
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


def main():
    # language detection is slow so i saved it for tests
    #df = pd.read_csv('original_data.csv')
    #df = delete_not_english(df)
    #df.to_csv('en_data.csv')

    ### test ###
    df = pd.read_csv('en_data.csv')
    df = clean_df(df)
    print(df['text'][0])


if __name__ == "__main__":
    main()
