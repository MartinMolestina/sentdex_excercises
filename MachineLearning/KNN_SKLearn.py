# Importing required libraries
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pandas as pd

# Reading csv as a panda's dataframe
df = pd.read_csv('datasets/breast-cancer-wisconsin-data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Defining features and objective
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# Dividinge dtaset between train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the Machine Learning algorithm
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Testing the accuracy of the algorithm with test data
accuracy = clf.score(X_test, y_test)
print(f"KNN's accuracy: {round(accuracy,3)}")

# Using the algorithm to classify a random patient
example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(f"KNN's classifies {example_measures[0]} as class {prediction}")