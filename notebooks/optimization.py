import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data/clean_car_price.csv")
print(df.shape)
#separer les variables de trainement et y la variable cible
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

#entrainement et test split

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("dimensions X_train :", X_train.shape)
print("dimensions X_test :", X_test.shape)

print("Dimensions du dataset :")
print(df.shape)

print("Colonnes du dataset :")
print(df.columns)

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

#preprocessing
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

#random forest
rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42
    ))
])


#xgboost

xgb_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBRegressor(
        random_state=42
    ))
])


# evaluation de random forest et xgboost

baseline_results = []


models = {
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}


for name, model in models.items():

    print(f"\nEntraînement du modèle de base : {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    baseline_results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


baseline_df = pd.DataFrame(baseline_results)

print(" les metriques des deux modeles avant optimization")
print(baseline_df)


#parametres de random forest , gridSearch pour random forest
rf_param_grid ={
    "model_n_estimators":[100,200],
    "model_max_depth":[None,10, 20],
    "model_min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)


print("optimization de random forest")

rf_grid.fit(X_train, y_train)

print("Meilleurs paramètres Random Forest :")
print(rf_grid.best_params_)

print("Meilleur score CV :")
print(rf_grid.best_score_)


#gridsearch pour xgboost

xgb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 6],
    "model__subsample": [0.8, 1.0]
}


xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)


print("optimization de xgboost")

xgb_grid.fit(X_train, y_train)

print("Meilleurs paramètres XGBoost :")
print(xgb_grid.best_params_)

print("Meilleur score CV :")
print(xgb_grid.best_score_)


# evaluation des modeles apres optimization

optimized_models = {
    "Random Forest Optimisé": rf_grid.best_estimator_,
    "XGBoost Optimisé": xgb_grid.best_estimator_
}


optimized_results = []


for name, model in optimized_models.items():

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    optimized_results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })


optimized_df = pd.DataFrame(optimized_results)


print("les metriques apres optimization")
print(optimized_df)


#comparaison avant,apres

print("COMPARAISON")

print("Avant optimisation :")
print(baseline_df)

print("Apres optimisation :")
print(optimized_df)