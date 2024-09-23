import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# upload the dataset

pcos_data_survey = pd.read_csv("CLEAN- PCOS SURVEY SPREADSHEET.csv")

# removing outliers

Q1 = pcos_data_survey["Height (in Cm / Feet)"].quantile(0.25)
Q3 = pcos_data_survey["Height (in Cm / Feet)"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
pcos_data_survey["Height (in Cm / Feet)"][pcos_data_survey["Height (in Cm / Feet)"]<lower_bound] = Q1

# splitting the data into features and target

X = pcos_data_survey.drop("Have you been diagnosed with PCOS/PCOD?",axis=1)
y = pcos_data_survey["Have you been diagnosed with PCOS/PCOD?"]

# Standardized the features

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# Splitting the features into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=18)

# traiing the SVC mdel

svc = SVC(kernel='rbf',C=1)
svc.fit(X_train,y_train)

# saving the SVC model and the scalar

joblib.dump(svc,"SVC.pkl")
joblib.dump(scalar,"scalar.pkl")
