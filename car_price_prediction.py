import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/car_price.csv")
print("********** shape ***********")
print(df.shape)

print("********** info ***********")
print(df.info())

print("********** head 4 ***********")
print(df.head(4))

print("********** describe ***********")
print(df.describe())

print("********** mediane de selling_price ***********")
print(df["selling_price"].median(numeric_only=True))

print(df["fuel"].value_counts())

print("*********tail*")
print(df.tail())

print("***les columns***")

print(df.columns)


print("***les types***")

print(df.dtypes)


plt.figure(figsize=(8, 5))
plt.xlabel("Kilométrage")
plt.ylabel("Fréquence")
plt.title("Distribution du kilométrage")
plt.hist(df["km_driven"], bins=30)
plt.show()
