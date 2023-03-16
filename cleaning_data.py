import pandas as pd
import re
from langdetect import detect
import numpy as np
import string
from unicodedata import category
import sys

def check_languages(df):
    langs = []
    for i in range(len(df['text'])):
        try:
            lang = detect(df['text'][i])
            langs.append(lang)
        except:
            print(i)
    return np.unique(np.array(langs))

def delete_stopwords(text):
    # TODO: write this function lol
    pass

def clean_text(text, punctuation_chars):
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
    return text

def clean_df(df):
    df = df.fillna('')
    punctuation_chars = [chr(i) for i in range(sys.maxunicode)
                         if category(chr(i)).startswith("P")]
    df['text'] = df['text'].map(lambda x: clean_text(x, punctuation_chars))
    df['title'] = df['title'].map(lambda x: clean_text(x, punctuation_chars))
    return df

def main():
    # test
    df = pd.read_csv('original_data.csv')
    df = clean_df(df)
    print(df['text'][1])

if __name__ == "__main__":
    main()