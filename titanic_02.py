# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn model validation and preprocessing imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from future_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score

# import models to try 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs_name = ["LogisticRegression", "SVC", "KNeighbors", "DecisionTree",
                "RandomForest", "GradientBoosting" ]

# Data import
titanic = pd.read_csv('train.csv')

# Drop useless columns and NaN values in the Embarked column
titanic.drop(columns = ["Ticket","Name","Cabin"], inplace=True)
titanic = titanic[titanic["Embarked"].isna()==False]

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Sex', 'Embarked']


# Pipeline to treat the datas
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_features)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

sex_pipeline = Pipeline([
        ('selector', DataFrameSelector(["Sex"])),
        ('sex_encoder', OrdinalEncoder()), #OneHotEncoder(sparse=False)
    ])

emb_pipeline = Pipeline([
        ('selector', DataFrameSelector(["Embarked"])),
        ('emb_encoder', OrdinalEncoder()), #OneHotEncoder(sparse=False)
    ])

data_preparation_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("sex_pipeline", sex_pipeline),
        ("emb_pipeline", emb_pipeline),
    ], n_jobs=1)

# Prepared data and labels for the fitting
X = data_preparation_pipeline.fit_transform(titanic)
y = titanic["Survived"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)

# Classificator to try 
clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier()) 

mean_clfs  = []
std_clfs = []

for name, classifier in zip(clfs_name, clfs):
    scores = cross_val_score(classifier, X_train, y_train, cv = 5, scoring="accuracy")
    print('---------------------------------')
    print(name, ':')
    print('---------------------------------')
    print('Mean: ', scores.mean())
    print('Std: ', scores.std())
    mean_clfs.append(scores.mean())
    std_clfs.append(scores.std())

mean_clfs_num = np.asarray(mean_clfs)
std_clfs_num = np.asarray(std_clfs)

# Visualization of the model scores
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('Results for each model')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
plt.errorbar(np.arange(1,mean_clfs_num.size+1),mean_clfs_num,yerr=2*std_clfs_num,fmt='o',
            elinewidth=1, capsize=3, capthick=1)
plt.ylim((0,1))
plt.xticks(np.arange(1,mean_clfs_num.size+1), clfs_name, rotation=45)
ax1.set_xlabel('Classificators', labelpad=15, fontsize=15)
ax1.set_ylabel('Mean score', labelpad=15, fontsize=15)


