import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

df = pd.read_csv("data/car_price.csv")   
#les premiers lignes
print("Premières lignes :")
print(df.head())

#les derniers lignes
print("Dernières lignes :")
print(df.tail())

# Dimensions (lignes,column)
print("Dimensions du dataset :")
print(df.shape)

# Noms des colonnes
print("Colonnes :")
print(df.columns)

# Types des colonnes
print("Types des colonnes :")
print(df.dtypes)

# Infos generales
print("Infos generales sont :")
df.info()


# Statistiques (moyen , max,min)
print("Statistiques descriptives :")
print(df.describe())

# median des columns numeriques
print("Médianes :")
print(df.median(numeric_only=True))

# #frequenses selon column
print("frequence de fuel")
print(df["fuel"].value_counts())

print("frequence des anneess")
print(df["year"].value_counts())

print("frequence des transmissions")
print(df["transmission"].value_counts())

print("frequence des owner")
print(df["owner"].value_counts())

#histogramme du km
plt.figure(figsize=(8,5))
plt.hist(df["km_driven"],bins=30)
plt.title("histogramme de kilometrage")
plt.xlabel("kilometrage")
plt.ylabel("frequence")
plt.show()


# HISTOGRAMME DU PRIX
plt.figure(figsize=(8,5))
plt.hist(df["selling_price"],bins=30)
plt.xlabel("prix de vent")
plt.ylabel("frequence")
plt.title("histogramme de prix de vente")
plt.show()

#impact de year sur le prix

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="year", y="selling_price")
plt.xlabel("year")
plt.ylabel("selling price")
plt.title("impact de annee sur le prix de vente")
plt.show()


# impact de km_driving sur selling_price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="km_driven", y="selling_price")
plt.xlabel("kelometrage")
plt.ylabel("prix de vente")
plt.title("impact de kilometrage sur le prix de vente")
plt.show()


#matrice de correlation
print("la matrice de correlation est:")
corr = df[["year", "km_driven", "selling_price"]].corr()
print(corr)

#heatmap
plt.figure(figsize=(8,5))
sns.heatmap(corr,annot=True, cmap="coolwarm")
plt.title("matrice de correlation")
plt.show()
 
#PAIRPLOT
sns.pairplot(df[["year", "km_driven", "selling_price"]])
plt.show()   