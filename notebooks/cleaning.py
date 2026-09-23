import pandas as pd
import numpy as np

df = pd.read_csv("data/car_price.csv")   
 

#valeurs manquants
print("Valeurs manquantes par colonne")
print(df.isnull().sum())  
  
# (imputation par la médiane pour les numériques
if df["year"].isna().sum() > 0:
    median_year = df["year"].median()
    df["year"] = df["year"].fillna(median_year)
    
if df["km_driven"].isna().sum() > 0:
    median_kmDriven= df["km_driven"].median()
    df["km_driven"]=df["km_driven"].fillna(median_kmDriven)
    
#imputation par le mode pour les catégoriques).
if df["fuel"].isna().sum()>0:
    modFuel=df["fuel"].mode()[0]
    df["fuel"]=df["fuel"].fillna(modFuel)


if df["seller_type"].isnull().sum() > 0:
    mode_seller_type = df["seller_type"].mode()[0]
    df["seller_type"] = df["seller_type"].fillna(mode_seller_type)

if df["owner"].isnull().sum() > 0:
    mode_owner = df["owner"].mode()[0]
    df["owner"] = df["owner"].fillna(mode_owner)   

# verifier si remplis
print("Valeurs manquantes après traitement :")
print(df.isnull().sum())    

#RECHERCHE DOUBLONS
print("voici le nombre des doublons dans columns")
print(df.duplicated().sum())
#supprimer les doublons
df.drop_duplicates()
 #verifier si dropped
print("voici le nombre des doublons apres drop")
print(df.duplicated().sum())


# 20. DIMENSIONS FINALES
print("Dimensions finales du dataset :")
print(df.shape)