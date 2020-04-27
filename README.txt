Link to dataset used: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
Link to video presentation: https://youtu.be/ypEVia8Vvfw

In order to run preprocessing.py, you need the following files in the same directory:
Fake.csv, Real.csv
then run the following script: python preprocessing.py 
	Output file should be news_Stemmed.csv

In order to run binary_data_creation.py, you need the following file in the same directory:
news_Stemmed.csv (outputted by preprocessing.py)
then run the following script: python binary_data_creation.py
	Output files should be BINARY_xTrain.csv, BINARY_xTest.csv, BINARY_yTrain.csv, BINARY_yTest.csv

In order to run tfidf_data_creation.py, you need the following file in the same directory:
news_Stemmed.csv (outputted by preprocessing.py)
then run the following script: python tfidf_data_creation.py
	Output files should be TFIDF_xTrain.csv, TFIDF_xTest.csv, TFIDF_yTrain.csv, TFIDF_yTest.csv

In order to run count_data_creation.py, you need the following file in the same directory:
news_Stemmed.csv (outputted by preprocessing.py)
then run the following script: python count_data_creation.py
	Output files should be COUNT_xTrain.csv, COUNT_xTest.csv, COUNT_yTrain.csv, COUNT_yTest.csv

In order to run sentiment_analysis.py, you need the following files in the same directory:
Fake.csv, Real.csv
then run the following script: python sentiment_analysis.py
	Outputs the figures used on the report

In order to run model.py, you need the following files in the same directory:
COUNT_xTrain.csv, COUNT_xTest.csv, COUNT_yTrain.csv, COUNT_yTest.csv, TFIDF_xTrain.csv, TFIDF_xTest.csv, TFIDF_yTrain.csv, TFIDF_yTest.csv, BINARY_xTrain.csv, BINARY_xTest.csv, BINARY_yTrain.csv, BINARY_yTest.csv
then run the following script: python sentiment_analysis.py
	output will be the accuracy of models on data