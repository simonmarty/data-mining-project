import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize



#this function uses the train data to create a dictionary, which contains words that are found in 100 or more news articles.
def build_vocab_map(df):

	words = {}

	#this for loop iterates through every sample in the dataframe
	for x in range(len(df.index)):
		wordList = df.values[x, 0].split()#for every article, create a list containing every word
		wl_no_duplicates = []
		for word in wordList: #This for loop deletes the duplicate words in a news article
			if word not in wl_no_duplicates:
				wl_no_duplicates.append(word)
		
		for item in wl_no_duplicates:#makes a dictionary with the number of news articles that contain that word
			if item not in words:
				words[item] =1
			else:
				words[item]+=1

	#This next part of the code deletes any words from the dictionary that are seen less in less than 100 news articles
	delete = [key for key, val in words.items() if val<100]
	for key in delete:
			del words[key]

	return words

#splits the news data into train and test in order to be pushed into a training model
def split_Data(x, y): 
	new_x = x.to_numpy()
	new_y = y.to_numpy()

	xTrain, xTest, yTrain, yTest = train_test_split(new_x, new_y, test_size =.3)

	xTrain = pd.DataFrame(xTrain)
	xTest = pd.DataFrame(xTest)
	yTrain = pd.DataFrame(yTrain)
	yTest = pd.DataFrame(yTest)

	return xTrain, yTrain, xTest, yTest

#transform Text attribute into binary counter attributes. Each word in the dictiony is transformed into an attribute
def construct_binary(df, dictionary):
	vectorizer = CountVectorizer(vocabulary = dictionary.keys(), binary = True)
	x = vectorizer.fit_transform(df.values[:, 0])
	df2 = pd.DataFrame(x.toarray())

	return df2

#normalzed the data using min-max scaler
def normalize_dataframe(df):

    x = df.to_numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(df_scaled)
    return df


def main():

	parser = argparse.ArgumentParser()

	#adds parameters for input and output files. these have a "default' name so that no parameters have to be added at the command prompt
	parser.add_argument("--news",
						default="news_Stemmed.csv",
						help="filename of the input data")
	parser.add_argument("--BINARY_xTrain",
						default = 'BINARY_xTrain.csv')
	parser.add_argument("--BINARY_yTrain",
						default = 'BINARY_yTrain.csv')
	parser.add_argument("--BINARY_xTest",
						default = 'BINARY_xTest.csv')
	parser.add_argument("--BINARY_yTest",
						default = 'BINARY_yTest.csv')

	args = parser.parse_args()


	#converts the file into a pandas dataframe, and fills null values with an empty string
	news = pd.read_csv(args.news)
	news = news.fillna("")

	xTrain, yTrain, xTest, yTest = split_Data(news['text'], news['label'])


	dictionary = build_vocab_map(xTrain)

	#create the train and test data for binary counter, and normalize it
	binary_dataset_train = construct_binary(xTrain, dictionary)
	binary_dataset_test = construct_binary(xTest, dictionary)
	binary_dataset_train = normalize_dataframe(binary_dataset_train)
	binary_dataset_test = normalize_dataframe(binary_dataset_test)

	#export the train and test dataframes to csv files
	binary_dataset_train.to_csv(args.BINARY_xTrain, index = False)
	binary_dataset_test.to_csv(args.BINARY_xTest, index = False)
	yTrain.to_csv(args.BINARY_yTrain, index = False)
	yTest.to_csv(args.BINARY_yTest, index = False)



if __name__ == '__main__':
	main()
