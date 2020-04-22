import argparse
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def Multinomial_NB(xTrain, yTrain, xTest, yTest):
	clf = MultinomialNB()
	clf.fit(xTrain, yTrain)
	return clf.score(xTest, yTest)

def logistic_regression(xTrain, yTrain, xTest, yTest):
	clf = LogisticRegression(solver='lbfgs')
	clf.fit(xTrain, yTrain)
	return clf.score(xTest, yTest)
def file_to_numpy(filename):
	"""
	Read an input file and convert it to numpy
	"""
	df = pd.read_csv(filename)
	return df.to_numpy()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tfidf_xTrain",
						help="filename for features of the tfidf training data",
						default = "TFIDF_xTrain.csv")

	parser.add_argument("--tfidf_yTrain",
						help="filename for labels associated with tfidf training data",
						default = "TFIDF_yTrain.csv")

	parser.add_argument("--tfidf_xTest",
						help="filename for features of the tfidf test data",
						default = "TFIDF_xTest.csv")

	parser.add_argument("--tfidf_yTest",
						help="filename for labels associated with the tfidf test data",
						default = "TFIDF_yTest.csv")



	args = parser.parse_args()

	tfidf_xTrain = file_to_numpy(args.tfidf_xTrain)
	tfidf_yTrain = file_to_numpy(args.tfidf_yTrain)
	tfidf_yTrain = tfidf_yTrain.ravel()
	tfidf_xTest = file_to_numpy(args.tfidf_xTest)
	tfidf_yTest = file_to_numpy(args.tfidf_yTest)
	tfidf_yTest = tfidf_yTest.ravel()

	tfidf_accuracy_NB = Multinomial_NB(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest)
	print("Accuracy for tfidf dataset in Naive Bayes: ", tfidf_accuracy_NB)

	tfidf_accuracy_LG = logistic_regression(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest)
	print("Accuracy for tfidf dataset in Logistic Regression: ", tfidf_accuracy_LG)

if __name__ == '__main__':
	main()