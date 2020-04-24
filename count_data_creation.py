import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


def build_vocab_map(df):
    words = {}

    # this for loop iterates through every sample in the dataframe
    for x in range(len(df.index)):
        wordList = df.values[x, 0].split()  # for every article, create a list containing every word
        wl_no_duplicates = []
        for word in wordList:  # This for loop deletes the duplicate words in a news article
            if word not in wl_no_duplicates:
                wl_no_duplicates.append(word)

        for item in wl_no_duplicates:  # makes a dictionary with the number of news articles that contain that word
            if item not in words:
                words[item] = 1
            else:
                words[item] += 1

    # This next part of the code deletes any words from the dictionary that are seen less in less than 100 news articles
    delete = [key for key, val in words.items() if val < 100]
    for key in delete:
        del words[key]

    return words


def split_data(x, y):
    new_x = x.to_numpy()
    new_y = y.to_numpy()

    xTrain, xTest, yTrain, yTest = train_test_split(new_x, new_y, test_size=.3)

    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)
    yTrain = pd.DataFrame(yTrain)
    yTest = pd.DataFrame(yTest)

    return xTrain, yTrain, xTest, yTest


def construct_count(df, dictionary):
    vectorizer = CountVectorizer(vocabulary=dictionary.keys(), binary=False)
    x = vectorizer.fit_transform(df.values[:, 0])
    df2 = pd.DataFrame(x.toarray())

    return df2


def normalize_dataframe(df):
    x = df.to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(df_scaled)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news",
                        default="news_Stemmed.csv",
                        help="filename of the input data")
    parser.add_argument("--COUNT_xTrain",
                        default='COUNT_xTrain.csv')
    parser.add_argument("--COUNT_yTrain",
                        default='COUNT_yTrain.csv')
    parser.add_argument("--COUNT_xTest",
                        default='COUNT_xTest.csv')
    parser.add_argument("--COUNT_yTest",
                        default='COUNT_yTest.csv')

    args = parser.parse_args()

    news = pd.read_csv(args.news)
    news = news.fillna("")

    xTrain, yTrain, xTest, yTest = split_Data(news['text'], news['label'])

    dictionary = build_vocab_map(xTrain)

    count_dataset_train = construct_count(xTrain, dictionary)
    count_dataset_test = construct_count(xTest, dictionary)
    count_dataset_train = normalize_dataframe(count_dataset_train)
    count_dataset_test = normalize_dataframe(count_dataset_test)

    count_dataset_train.to_csv(args.COUNT_xTrain, index=False)
    count_dataset_test.to_csv(args.COUNT_xTest, index=False)
    yTrain.to_csv(args.COUNT_yTrain, index=False)
    yTest.to_csv(args.COUNT_yTest, index=False)


if __name__ == '__main__':
    main()
