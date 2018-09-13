# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from future_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

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
        ('imputer', Imputer(strategy="most_frequent")),
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

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("sex_pipeline", sex_pipeline),
        ("emb_pipeline", emb_pipeline),
    ])


X = full_pipeline.fit_transform(titanic)
y = titanic["Survived"].values


# Random Tree Classifier
forest_clf = RandomForestClassifier()

param_grid = [
    {'n_estimators': list(range(5,18)),'max_depth': list(range(2,15))},
]


grid_search_forest = GridSearchCV(forest_clf, param_grid, cv=5, scoring='accuracy')

grid_search_forest.fit(X, y)
print("Random Forest Classifier: ")
print(grid_search_forest.best_params_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

forest_model = grid_search_forest.best_estimator_
forest_model.fit(X_train, y_train)

y_pred_train = forest_model.predict(X_train)
y_pred_test = forest_model.predict(X_test)
print("Accuracy score on train set", accuracy_score(y_pred_train, y_train))
print("Accuracy score on test set", accuracy_score(y_pred_test, y_test))

forest_scores = cross_val_score(forest_model, X, y, cv=10, scoring='accuracy')
print(forest_scores)
print('Mean: ', forest_scores.mean())
print('Standard Deviation: ', forest_scores.std())   

give_solution = False

if give_solution:

    titanic_test = pd.read_csv('test.csv')

    X_test = full_pipeline.transform(titanic_test)

    y_predicted = forest_model.predict(X_test).reshape(-1,1)
    pass_id = titanic_test["PassengerId"].values.reshape(-1,1)

    out = np.c_[pass_id, y_predicted]
    output = pd.DataFrame(out, columns = ["PassengerId", "Survived"])

    output.to_csv(path_or_buf="./Solution.csv", index=False)