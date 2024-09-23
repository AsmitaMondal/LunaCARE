import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# upload the dataset

pcos_data = pd.read_excel("PCOS_data_without_infertility_new.xlsx")

# removing outliers

pcos_data["RBS(mg/dl)"][pcos_data["RBS(mg/dl)"] > 140] = 140
pcos_data["RBS(mg/dl)"][pcos_data["RBS(mg/dl)"] < 70] = 70
pcos_data["FSH/LH"][pcos_data["FSH/LH"] > 5] = 5
pcos_data["TSH (mIU/L)"][pcos_data["TSH (mIU/L)"] > 5]  = 5
pcos_data["TSH (mIU/L)"][pcos_data["TSH (mIU/L)"] < 0.4]  = 0.4

# Splitting the data into features and target

X = pcos_data.loc[:,['Cycle(R/I)','Weight gain(Y/N)','hair growth(Y/N)','Skin darkening (Y/N)','Pimples(Y/N)','Fast food (Y/N)','Follicle No. (L)','Follicle No. (R)','Avg. F size (L) (mm)','Avg. F size (R) (mm)','TSH (mIU/L)','PRL(ng/mL)','BMI','FSH/LH','RBS(mg/dl)']]
y = pcos_data['PCOS (Y/N)']

# missing value imputation
X['Fast food (Y/N)'].fillna(X['Fast food (Y/N)'].mode()[0], inplace= True)

# Standardized the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the features into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=18)

# training the Gradient Boosting Classifier Model

gradient_boosting = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50)
gradient_boosting.fit(X_train,y_train)

# Saving the model and the scalar

joblib.dump(gradient_boosting,'gradient_boosting_model.pkl')
joblib.dump(scaler,'scaler.pkl')



