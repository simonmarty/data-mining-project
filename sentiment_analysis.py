import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk


def sentiment_analysis(df):
    x = df.to_numpy()
    list_of_subjectivity = []
    list_of_polarity = []
    for item in x:
        string = TextBlob(item)
        list_of_subjectivity.append(string.sentiment.subjectivity)
        list_of_polarity.append(string.sentiment.polarity)
    df2 = pd.DataFrame(data=list_of_subjectivity)
    df3 = pd.DataFrame(data=list_of_polarity)
    return df2, df3


def polarity_histogram(df, title):
    plt.hist(df)
    plt.xlabel("Polarity")
    plt.ylabel('frequency')
    plt.title(title)
    plt.show()


def subjectivity_histogram(df, title):
    plt.hist(df)
    plt.xlabel("Subjectivity")
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


def word_cloud_maker(words):
    wordcloud = WordCloud(width=3000, height=2000, background_color='black',
                          stopwords=set(nltk.corpus.stopwords.words("english"))).generate(str(words))
    fig = plt.figure(figsize=(40, 30), facecolor='k', edgecolor='k')
    plt.imshow(wordcloud)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_news"
                        , default="True.csv")
    parser.add_argument("--fake_news"
                        , default="Fake.csv")

    args = parser.parse_args()

    real = pd.read_csv(args.real_news)
    real = real.fillna("")
    fake = pd.read_csv(args.fake_news)
    fake = fake.fillna("")
    real['subjectivity for real text'], real['polarity for real text'] = sentiment_analysis(real['text'])
    fake['subjectivity for fake text'], fake['polarity for fake text'] = sentiment_analysis(fake['text'])

    real['subjectivity for real titles'], real['polarity for real titles'] = sentiment_analysis(real['title'])
    fake['subjectivity for fake titles'], fake['polarity for fake titles'] = sentiment_analysis(fake['title'])

    polarity_histogram(real['polarity for real text'], "Polarity Distribution of Real News")
    polarity_histogram(real['polarity for real titles'], "Polarity Distribution of Real News Titles")
    polarity_histogram(fake['polarity for fake titles'], "Polarity Distribution of Fake News Titles")
    polarity_histogram(fake['polarity for fake text'], "Polarity Distribution of Fake News")

    subjectivity_histogram(real['subjectivity for real text'], "Subjectivity Distribution of Real News")
    subjectivity_histogram(real['subjectivity for real titles'], "Subjectivity Distribution of Real News Titles")
    subjectivity_histogram(fake['subjectivity for fake titles'], "Subjectivity Distribution of Fake News Titles")
    subjectivity_histogram(fake['subjectivity for fake text'], "Subjectivity Distribution of Fake News")
    polarity = pd.concat(
        [real['polarity for real text'], real['polarity for real titles'], fake['polarity for fake text'],
         fake['polarity for fake titles']], axis=1)
    subjectivity = pd.concat(
        [real['subjectivity for real text'], real['subjectivity for real titles'], fake['subjectivity for fake text'],
         fake['subjectivity for fake titles']], axis=1)

    boxplot = subjectivity.boxplot(column=['subjectivity for real text', 'subjectivity for fake text'])
    boxplot.set_title('Subjectivity Distribution of News Articles')
    plt.show()

    boxplot2 = subjectivity.boxplot(column=['subjectivity for real titles', 'subjectivity for fake titles'])
    boxplot2.set_title('Subjectivity Distribution of News Titles')
    plt.show()

    boxplot3 = polarity.boxplot(column=['polarity for real text', 'polarity for fake text'])
    boxplot3.set_title('Polarity Distribution of News Articles')
    plt.show()

    boxplot4 = polarity.boxplot(column=['polarity for real titles', 'polarity for fake titles'])
    boxplot4.set_title('Polarity Distribution of News Titles')
    plt.show()

    fake_words = fake.text.values
    real_words = real.text.values
    real_titles = real.title.values
    fake_titles = fake.title.values

    word_cloud_maker(fake_words)
    word_cloud_maker(real_words)
    word_cloud_maker(fake_titles)
    word_cloud_maker(real_titles)


if __name__ == '__main__':
    main()
