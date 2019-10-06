import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.transforms
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn



bankdata = pd.read_csv("/home/elissonriller/comp_inteligence/svm/data_banknote_authentication.txt", header=None)
#print(bankdata.shape)
#print(bankdata.head())

x = bankdata.drop(4, axis=1)
y = bankdata[4]

#print(x)
#print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

