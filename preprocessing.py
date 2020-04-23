import argparse
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
def preprocess(real, fake):
	# unite fake and real news
	df = pd.concat([real, fake], axis=0)
	# shuffle the rows of the dataframe
	df = df.sample(frac=1).reset_index(drop=True)
	# drop unessesary data
	df = df.drop(columns=['title', 'subject', 'date'])
	df = df.fillna("")
	return df


def text_stemming(df):
    ps = PorterStemmer()
    x = df.to_numpy()
    y = []
    for item in x:
        x2 = word_tokenize(item.lower())
        s = ""
        for item2 in x2:
            s += ps.stem(item2)
            s += " "
        y.append(s)
    print(x)
    print(y)
    return pd.DataFrame(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_news"
                        , default="True.csv")
    parser.add_argument("--fake_news"
                        , default="Fake.csv")
    parser.add_argument('news')

    args = parser.parse_args()

    real = pd.read_csv(args.real_news)
    fake = pd.read_csv(args.fake_news)

    news = preprocess(real, fake)

    news['text'] = text_stemming(news['text'])

    news.to_csv(args.news, index=False)


if __name__ == '__main__':
    main()
