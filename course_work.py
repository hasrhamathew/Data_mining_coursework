# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:47:53 2023

@author: harsha
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sbn

# Reading the file
file = pd.read_csv('mushroom_data.csv')
print(file.shape)
print(file.dtypes)
# Converting categorical data into numerical data
enc = LabelEncoder()
for col in file.columns:
    enc.fit(file[col])
    file[col] = enc.transform(file[col])

print(file.dtypes)
# Checking for any null values
print(file.isnull().any())
# Deleting all the duplicate values if present
file.drop_duplicates()
# Since veil types is same for every row we can delete this coloumn
# Deleting the Veil_type column
file.drop("Veil_type", axis=1, inplace=True)
print(file.shape)

# Now we can create our x and y
# x is all the attributes
x = file.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
# y is our target variable which is Type
y = file.iloc[:, 0]

# Creating train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,
                                                    random_state=0)
# Scaling the data
stnd_scaler = StandardScaler()
x_train = stnd_scaler.fit_transform(x_train)
x_test = stnd_scaler.transform(x_test)

# Model creating
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)

# Predicting the test result
y_pred = gaussian.predict(x_test)

# Finding the accuracy score
accuracy_score = accuracy_score(y_test, y_pred)
print("\nAccuracy Score : ", str(round(accuracy_score, 3)))

# Creating a confusion matrix to find where our model is predicting correctly
# and where it's not
confusion_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix : ", confusion_matrix)

# Representing the confusion matrix in a heatmap for better understanding
plt.figure(figsize=(6, 6))
x_labels = ["Edible", "Poisonous"]
y_labels = ["Edible", "Poisonous"]
sbn.heatmap(confusion_matrix, annot=True, fmt=".0f",
            linewidth=.1, cmap="crest", vmin=50, vmax=750,
            xticklabels=x_labels, yticklabels=y_labels)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Value", fontsize=13)
plt.ylabel("True Value", fontsize=13)
plt.savefig("heatmap.png")
plt.show()
