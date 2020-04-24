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

    args = parser.parse_args()

    tfidf_xTrain = file_to_numpy(args.tfidf_xTrain)
    tfidf_yTrain = file_to_numpy(args.tfidf_yTrain)
    tfidf_yTrain = tfidf_yTrain.ravel()
    tfidf_xTest = file_to_numpy(args.tfidf_xTest)
    tfidf_yTest = file_to_numpy(args.tfidf_yTest).ravel()

    tfidf_accuracy_nb = train(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest, model=MultinomialNB())
    print("Accuracy for tfidf dataset in Multinomial Naive Bayes: ", tfidf_accuracy_nb)

    tfidf_accuracy_lg = train(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest,
                              model=LogisticRegression(solver='lbfgs'))
    print("Accuracy for tfidf dataset in Logistic Regression: ", tfidf_accuracy_lg)

    tfidf_accuracy_bi = train(tfidf_xTrain, tfidf_yTrain, tfidf_xTest, tfidf_yTest, model=BernoulliNB())
    print("Accuracy for tfidf dataset in Bernoulli Naive Bayes: ", tfidf_accuracy_bi)


if __name__ == '__main__':
    main()
