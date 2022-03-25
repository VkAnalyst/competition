import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('Data/ready_to_train.csv')

X = df.drop('winner', axis = 1)
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

rfc = RandomForestClassifier()
model = rfc.fit(X_train, y_train)

rfc_predict = model.predict(X_test)
#print(accuracy_score(y_test, rfc_predict))
pickle.dump(model, open('model.pkl', 'wb'))