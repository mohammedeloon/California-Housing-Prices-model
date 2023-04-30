import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
         housing_cat_1hot = self.encoder.transform(X)
         return housing_cat_1hot.toarray()
class HousingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income']
        self.cat_attribs = ['ocean_proximity']
        self.num_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.num_attribs)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])
        self.cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.cat_attribs)),
            ('encoder', CategoricalEncoder())
        ])
        self.full_pipeline = ColumnTransformer([
            ('num', self.num_pipeline, self.num_attribs),
            ('cat', self.cat_pipeline, self.cat_attribs),
        ])
    def fit_transform(self, X, y=None):
        return pd.DataFrame(self.full_pipeline.fit_transform(X))

# Load the housing data
housing = pd.read_csv('datasets/housing/housing_predictors.csv')

# Preprocess the housing data
housing_prepared = HousingPreprocessor().fit_transform(housing)
housing_prepared.to_csv('datasets/housing/housing_prepared.csv')
housing_prepared


