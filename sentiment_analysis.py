import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import nltk

def sentiment_analysis(df):
	x = df.to_numpy()
	list_of_subjectivity= []
	list_of_polarity = []
	for item in x:
		string = TextBlob(item)
		list_of_subjectivity.append(string.sentiment.subjectivity)
		list_of_polarity.append(string.sentiment.polarity)
	df2 = pd.DataFrame(data = list_of_subjectivity)
	df3 = pd.DataFrame(data = list_of_polarity)
	return df2, df3

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--real_news"
						, default = "True.csv")
	parser.add_argument("--fake_news"
						, default = "Fake.csv")


	args = parser.parse_args()

	real = pd.read_csv(args.real_news)
	real = real.fillna("")
	fake = pd.read_csv(args.fake_news)
	fake = fake.fillna("")
	real['subjectivity'], real['polarity'] = sentiment_analysis(real['text'])
	fake['subjectivity'], fake['polarity'] = sentiment_analysis(fake['text'])

	plt.hist(real['polarity'])
	plt.xlabel("Polarity")
	plt.ylabel('frequency')
	plt.title("Polarity Distribution of Real News")
	plt.show()

	plt.hist(real['subjectivity'])
	plt.xlabel("Subjectivity")
	plt.ylabel('Frequency')
	plt.title("Subjectivity Distribution of Real News")
	plt.show()

	plt.hist(fake['polarity'])
	plt.xlabel("Polarity")
	plt.ylabel('Frequency')
	plt.title("Polarity Distribution of Fake News")
	plt.show()

	plt.hist(fake['subjectivity'])
	plt.xlabel("Subjectivity")
	plt.ylabel('frequency')
	plt.title("Subjectivity Distribution of Fake News")
	plt.show()

	boxplot = real.boxplot(column = ['subjectivity', 'polarity'])
	boxplot.set_title('Sentiment Analysis for Real News')

	boxplot2 = fake.boxplot(column = ['subjectivity', 'polarity'])
	boxplot2.set_title('Sentiment Analysis for Fake News')
	plt.show()

	fake_words = fake.text.values
	real_words = real.text.values
	wordcloud = WordCloud(width = 3000,	height = 2000, background_color = 'black', stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(fake_words))
	fig = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')	
	plt.imshow(wordcloud)
	

	wordcloud2 = WordCloud(width = 3000, height = 2000, background_color = 'black', stopwords = set(nltk.corpus.stopwords.words("english"))).generate(str(real_words))
	fig = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')	
	plt.imshow(wordcloud2)
	plt.show()
	

if __name__ == '__main__':
	main()


