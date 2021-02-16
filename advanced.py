# Importing needed modules
from sklearn.metrics import classification_report
from utility import load_data_from_csv
import numpy as np
import pandas as pd


#Setting the file names needed to variables
testcsv = 'reviews_Video_Games_test.csv'

#Loading the needed files
testFeatureNames, testInstance, testLabels = load_data_from_csv(testcsv)
df_sentiment_lexicon = pd.read_csv('Games_senti_lexicon.tsv', delimiter='\t', header=None)
sentiment_words = list(df_sentiment_lexicon[0])
sentiment_scores = list(df_sentiment_lexicon[1])

def sentimentLexicon():
  #Open the needed files
  file = open("results.txt", 'w')
  df_test_data = pd.read_csv('reviews_Video_Games_test.csv', header=0)

  for feature in df_test_data.columns: 
  # if feature not in sentiment_words list then remove it from both training and test data
    if feature not in sentiment_words:
      df_test_data = df_test_data.drop(feature, axis=1)
  
  #convert the instances to a numpy array
  npArray = df_test_data.values

  #loop through each instance and if the word is in the instance add/subtract
  #the score of the word, then save it to a file
  for instance in npArray:
    instanceScore = 0
    for index, entry in enumerate(instance):
      if entry > 0:
        for count in range(entry):
          word = df_test_data.columns[index]
          wordLoc = sentiment_words.index(str(word))
          instanceScore += sentiment_scores[wordLoc]
    file.write(str(instanceScore) + "\n")
  #Once finished file is closed
  file.close()
  
  #the file is opened and read, each review is check if it is positive its saved as a 1
  #if negative then a 0. A classifiaction report is then ran on the results.
  file = open("results.txt", 'r')
  file = file.readlines()
  results = []
  for line in file:
    if float(line) >= 0:
      results.append(1)
    elif float(line) < 0:
      results.append(0)
    
  print(classification_report(testLabels, results, digits=3))


if __name__ == "__main__":
  sentimentLexicon()
