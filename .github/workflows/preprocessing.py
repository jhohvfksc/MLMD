import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
def create_pipeline():
  #Define which features are numeric vs categorical
  numeric_features=['agr','income']
  categorical_features=['city','occupation']
  #Preprocessing for numerical data
