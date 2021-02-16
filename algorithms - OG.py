# Importing needed modules
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import numpy as np
import pandas as pd
import pandas.io.excel._openpyxl
from pandas.io.excel import ExcelWriter
from utility import load_data_from_csv
from utility import append_df_to_excel
import sys

#Setting the file names needed to variables
trainingcsv = 'reviews_Video_Games_training.csv'
testcsv = 'reviews_Video_Games_test.csv'
excelFile = 'allResults.xlsx'

#Loading the training data and the testing data
trainingFeatureNames, trainingInstance, trainingLabels = load_data_from_csv(trainingcsv)
testFeatureNames, testInstance, testLabels = load_data_from_csv(testcsv)

"""
User inputs the amount of values of K they want to run and 
the program runs for that many values of K
"""
def knn(loops):
  for modifier in range(int(loops)):
    print("loop number: ", modifier)
    classifier = KNeighborsClassifier(n_neighbors=modifier+1)
    classifier.fit(trainingInstance, trainingLabels)
    predictions = classifier.predict(testInstance)
    report = classification_report(testLabels, predictions, digits=3, output_dict=True)
    report['k Value'] = modifier+1
    pdFrame = pd.DataFrame(report)
    append_df_to_excel(excelFile, pdFrame, 'knn')

"""
User inputs if they want hard(Majority) or Soft(confidence)
and the amount of K-NN classifiers they want also uses
a decision tree
"""
def ensamble(type, value):
  kValue = int(value)

  ensambleEstimators = [('decisionTree', tree.DecisionTreeClassifier())]

  for value in range(kValue):
    ensambleEstimators.append(('knn'+str(value+1), KNeighborsClassifier(n_neighbors=value+1)))

  if(type == 'hard'):
    ensamble = VotingClassifier(estimators= ensambleEstimators, voting='hard')
  elif(type == 'soft'):
    ensamble = VotingClassifier(estimators= ensambleEstimators, voting='soft')
  
  ensamble.fit(trainingInstance, trainingLabels)
  predictions = ensamble.predict(testInstance)
  report = classification_report(testLabels, predictions, digits=3, output_dict=True)
  pdFrame = pd.DataFrame(report)
  append_df_to_excel(excelFile, pdFrame, 'ensamble'+type)

"""
User inputs the amount of K-NN classifiers to use.
stacks with a decision tree
"""
def stacking(value):
  kValue = int(value)

  stackingClass = [] 

  for value in range(kValue):
    stackingClass.append(KNeighborsClassifier(n_neighbors=value+1))

  metaclassifier = tree.DecisionTreeClassifier()
  stacking = StackingClassifier(classifiers=stackingClass, meta_classifier=metaclassifier)
  stacking.fit(trainingInstance, trainingLabels)
  predictions = stacking.predict(testInstance)
  report = classification_report(testLabels, predictions, digits=3, output_dict=True)
  pdFrame = pd.DataFrame(report)
  append_df_to_excel(excelFile, pdFrame, 'stacking')

"""
Naive Bayes Classifier
"""
def naiveBayes():
  classifier = GaussianNB()
  classifier.fit(trainingInstance, trainingLabels)
  predictions = classifier.predict(testInstance)
  report = classification_report(testLabels, predictions, digits=3, output_dict=True)
  pdFrame = pd.DataFrame(report)
  append_df_to_excel(excelFile, pdFrame, 'naiveBayes')

"""
Random Forest, user inputs then number of trees the amount
of random features per tree and the computational nodes
needed.
"""
def randomForest(estimators, features, nodes):
  """ 
  n_estimator: k-parameter (no of decision trees)
  max_feature: m-parameter (no of random features)
  n_jobs:  Number of threads to be used.
      if > 1 train decision in parallel (Doesn't affect F-measure)
  """
  estimators = int(estimators)
  features = int(features)
  nodes = int(nodes)

  classifier = RandomForestClassifier(n_estimators=estimators, max_features=features, n_jobs=nodes)
  classifier.fit(trainingInstance, trainingLabels)
  predictions = classifier.predict(testInstance)
  report = classification_report(testLabels, predictions, digits=3, output_dict=True)
  pdFrame = pd.DataFrame(report)
  append_df_to_excel(excelFile, pdFrame, 'randomForest')

"""
User defines the kernal used to run the data
  For non-linear SVM change kernel to:
  rbf, sigmoid.
  linear is just linear kernel
"""
def supportvectormachine(svmKernel):
  classifier = svm.SVC(kernel=svmKernel)
  classifier.fit(X=trainingInstance, y=trainingLabels)
  predictions = classifier.predict(testInstance)
  report = classification_report(testLabels, predictions, digits=3, output_dict=True)
  pdFrame = pd.DataFrame(report)
  append_df_to_excel(excelFile, pdFrame, 'svm'+svmKernel)

"""
Takes the users input from command line and runs the corresponding classifier
"""
if __name__ == "__main__":
    args = sys.argv[1:]
    if (args[0] == "knn"):
      knn(args[1])
    elif (args[0] == 'ensamble'):
      ensamble(args[1], args[2])
    elif (args[0] == 'stacking'):
      stacking(args[1])
    elif (args[0] == 'naive'):
      naiveBayes()
    elif (args[0] == 'forest'):
      randomForest(args[1], args[2], args[3])
    elif (args[0] == 'svm'):
      supportvectormachine(args[1])