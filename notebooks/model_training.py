import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df=pd.read_csv("data/clean_car_price.csv")
print("les dimensions de data:")
print(df.shape)
print("les column de data cleaning est:")
print(df.columns)

#separer x et y
X=df.drop(columns=["selling_price"])
y=df["selling_price"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

#preprocessing
numeric_features=["year","km_driven"]
categorical_featues=["fuel","seller_type","transmission","owner"]

numeric_transformer=StandardScaler()
categorical_transformer=OneHotEncoder(
     drop="first",
     handle_unknown="ignore"
)
preprocessor=ColumnTransformer(
    transformers=[
        ("nums",numeric_transformer,numeric_features),
        ("cat",categorical_transformer,categorical_featues)
    ]
)

#creation des quatre models
linearModel=LinearRegression()
rfModel=RandomForestRegressor()
svrModel=SVR()
xgboostModel=XGBRegressor()

#creation de pipeline
linearPipeline=Pipeline([
    ("preprocessing",preprocessor),
    ("model",linearModel)
])
rfPipeline=Pipeline([
    ("preprocessing",preprocessor),
    ("model",rfModel)
])

svrPipeline=Pipeline([
    ("preprocessing",preprocessor),
    ("model",svrModel)
])
xgboostPipeline=Pipeline([
    ("preprocessing",preprocessor),
    ("model",xgboostModel)
])

#les models dans dictionnaires
resultats=[]
models= {
    "linear regression":linearPipeline,
    "random forest":rfPipeline,
    "svr":svrPipeline,
    "xgboost ":xgboostPipeline
}
resultats=[]
for name,model in models.items():
    print(f"entrainement de:{name}")
    model.fit(X_train,y_train)

#predection
predictions={}
for name,model in models.items():
    predictions[name]=model.predict(X_test)

#calcul des RMSE,MAE , R^2
for name,y_pred in predictions.items():
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=mse**0.5
    r2=r2_score(y_test,y_pred)
    resultats.append({
      "Model":name,
      "RMSE":rmse,
      "MSE":mse,
      "R2":r2,
      "MAE":mae
    })
results_df = pd.DataFrame(resultats)
print(results_df)