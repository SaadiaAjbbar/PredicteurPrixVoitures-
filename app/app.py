import streamlit as st
import pandas as pd
import joblib

# 1. Charger le modele

model = joblib.load("models/random_forest_optimized.pkl")

# 2. Titre de lapplication

st.title("Predicteur de prix des voitures")

st.write(
    "Entrez les caracteristiques du vehicule "
    "pour obtenir une estimation de son prix."
)


# 3. Formulaire de saisie

st.header("Caracteristiques du vehicule")


year = st.number_input(
    "Annee du vehicule",
    min_value=1990,
    max_value=2026,
    value=2015,
    step=1
)

km_driven = st.number_input(
    "Kilometrage (km)",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
)


fuel = st.selectbox(
    "Carburant",
    ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
)


seller_type = st.selectbox(
    "Type de vendeur",
    ["Individual", "Dealer", "Trustmark Dealer"]
)


transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)


owner = st.selectbox(
    "Proprietaire",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

#  Bouton de prediction

if st.button("Estimer le prix"):

    # Creer les donnees du vehicule
    vehicle = pd.DataFrame({
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })


    # prediction
    prediction = model.predict(vehicle)


    
    predicted_price = prediction[0]


    # Afficher le resultat
    st.success(
        f"Prix estime : {predicted_price:,.0f} ₹"
    )

