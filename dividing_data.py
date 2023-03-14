import pandas as pd
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv('original_data.csv')
x = df[['title', 'text']]
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

# saving to files (you need to
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


x_test.to_csv('validation_data/x_test.csv', encoding='utf-8')
y_test.to_csv('validation_data/y_test.csv', encoding='utf-8')
print('validation data saved')

x_train_train.to_csv('train_data/x_train_train.csv', encoding='utf-8')
x_train_test.to_csv('train_data/x_train_test.csv', encoding='utf-8')
y_train_train.to_csv('train_data/y_train_train.csv', encoding='utf-8')
y_train_test.to_csv('train_data/y_train_test.csv', encoding='utf-8')
print('test data saved')

print('success! c:')
