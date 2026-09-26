import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

df = pd.read_csv("data/clean_car_price.csv")

X = df.drop(columns=["selling_price"])
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

numeric_features = [
    "year",
    "km_driven"
]
categorical_features = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

linear_model = LinearRegression()

rf_model = RandomForestRegressor()

svr_model = SVR()

linear_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", linear_model)
])
rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", rf_model)
])
svr_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", svr_model)
])