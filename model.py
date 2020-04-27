# Haris Ahmad, Luis Gomez Flores, Jack Frumkes, Simon Marty

import argparse
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def train(xTrain, yTrain, xTest, yTest, model):
    model.fit(xTrain, yTrain)
    return model.score(xTest, yTest)


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
                        default="TFIDF_xTrain.csv")

    parser.add_argument("--tfidf_yTrain",
                        help="filename for labels associated with tfidf training data",
                        default="TFIDF_yTrain.csv")

    parser.add_argument("--tfidf_xTest",
                        help="filename for features of the tfidf test data",
                        default="TFIDF_xTest.csv")

    parser.add_argument("--tfidf_yTest",
                        help="filename for labels associated with the tfidf test data",
                        default="TFIDF_yTest.csv")

    parser.add_argument("--binary_xTrain",
                        help="filename for features of the tfidf training data",
                        default="BINARY_xTrain.csv")

    parser.add_argument("--binary_yTrain",
                        help="filename for labels associated with tfidf training data",
                        default="BINARY_yTrain.csv")

    parser.add_argument("--binary_xTest",
                        help="filename for features of the tfidf test data",
                        default="BINARY_xTest.csv")

    parser.add_argument("--binary_yTest",
                        help="filename for labels associated with the tfidf test data",
                        default="BINARY_yTest.csv")

    parser.add_argument("--count_xTrain",
                        help="filename for features of the tfidf training data",
                        default="COUNT_xTrain.csv")

    parser.add_argument("--count_yTrain",
                        help="filename for labels associated with tfidf training data",
                        default="COUNT_yTrain.csv")

    parser.add_argument("--count_xTest",
                        help="filename for features of the tfidf test data",
                        default="COUNT_xTest.csv")

    parser.add_argument("--count_yTest",
                        help="filename for labels associated with the tfidf test data",
                        default="COUNT_yTest.csv")
    args = parser.parse_args()

    tfidf_xTrain = file_to_numpy(args.tfidf_xTrain)
    tfidf_yTrain = file_to_numpy(args.tfidf_yTrain)
    tfidf_yTrain = tfidf_yTrain.ravel()
    tfidf_xTest = file_to_numpy(args.tfidf_xTest)
    tfidf_yTest = file_to_numpy(args.tfidf_yTest).ravel()

    binary_xTrain = file_to_numpy(args.binary_xTrain)
    binary_yTrain = file_to_numpy(args.binary_yTrain)
    binary_yTrain = binary_yTrain.ravel()
    binary_xTest = file_to_numpy(args.binary_xTest)
    binary_yTest = file_to_numpy(args.binary_yTest).ravel()

    count_xTrain = file_to_numpy(args.count_xTrain)
    count_yTrain = file_to_numpy(args.count_yTrain)
    count_yTrain = count_yTrain.ravel()
    count_xTest = file_to_numpy(args.count_xTest)
    count_yTest = file_to_numpy(args.count_yTest).ravel()

    print("running naive bayes model on tfidf data:")
    tfidf_accuracy_nb = train(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest, model=MultinomialNB())
    print("Accuracy for tfidf dataset in Multinomial Naive Bayes: ", tfidf_accuracy_nb)
    print("running logistic regression model on tfidf data")
    tfidf_accuracy_lg = train(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest,
                              model=LogisticRegression(solver='lbfgs'))
    print("Accuracy for tfidf dataset in Logistic Regression: ", tfidf_accuracy_lg)

    print("")

    print("running naive bayes model on binary data:")
    binary_accuracy_bi = train(binary_xTrain, binary_yTrain, binary_xTest, binary_yTest, model=BernoulliNB())
    print("Accuracy for binary dataset in Bernoulli Naive Bayes: ", binary_accuracy_bi)

    print("running logistic regression on binary data")
    binary_accuracy_lg = train(binary_xTrain, binary_yTrain, binary_xTest, binary_yTest,
                                model=LogisticRegression(solver = 'lbfgs'))
    print("Accuracy for binary dataset in Multinomial Naive Bayes: ", binary_accuracy_lg)
    print("")

    print("running naive bayes momdel on count data:")
    count_accuracy_nb = train(count_xTrain, count_yTrain, count_xTest, count_yTest, model=MultinomialNB())
    print("Accuracy for count dataset in Multinomial Naive Bayes: ", count_accuracy_nb)
    print("running logistic regression on count data")
    count_accuracy_lg = train(count_xTrain, count_yTrain, count_xTest, count_yTest,
                              model=LogisticRegression(solver='lbfgs'))
    print("Accuracy for count dataset in Logistic Regression: ", count_accuracy_lg)


if __name__ == '__main__':
    main()
