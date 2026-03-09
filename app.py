
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =============================
# CONFIGURATION DE LA PAGE
# =============================
st.set_page_config(
    page_title='Credit Scoring App',
    layout='centered'
)

# =============================
# CHARGEMENT DU MODELE
# =============================
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# =============================
# INTERFACE UTILISATEUR
# =============================
st.title('Credit Scoring App')
st.subheader('Prédiction du risque de remboursement')
st.markdown('---')
st.write('Remplissez le formulaire ci-dessous pour évaluer le risque crédit d un client.')

# =============================
# FORMULAIRE DE SAISIE
# =============================
st.header('Informations du Client')

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age du client', min_value=18, max_value=75, value=35)
    credit_amount = st.number_input('Montant du crédit (EUR)', min_value=250, max_value=20000, value=5000, step=250)
    duration = st.slider('Durée du crédit (mois)', min_value=4, max_value=72, value=24)

with col2:
    sex = st.selectbox('Sexe', options=[0, 1],
        format_func=lambda x: {0: 'Femme', 1: 'Homme'}[x])
    job = st.selectbox('Type d emploi', options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: 'Sans emploi',
            1: 'Employé (CDD / Temps partiel)',
            2: 'Employé (CDI)',
            3: 'Cadre / Indépendant / Fonctionnaire'
        }[x])
    housing = st.selectbox('Logement', options=[0, 1, 2],
        format_func=lambda x: {
            0: 'Hébergé',
            1: 'Propriétaire',
            2: 'Locataire'}[x])

saving_accounts = st.selectbox('Compte épargne', options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: 'Aucun compte épargne',
        1: 'Épargne faible (< 100 euros)',
        2: 'Épargne modérée (100 - 500 euros)',
        3: 'Bonne épargne (500 - 1000 euros)',
        4: 'Épargne solide (> 1000 euros)'
    }[x])

checking_account = st.selectbox('Compte courant', options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: 'Aucun compte courant',
        1: 'Compte débiteur (solde négatif)',
        2: 'Solde faible (0 - 200 EUR)',
        3: 'Solde confortable (> 200 EUR)'
    }[x])

purpose = st.selectbox('But du crédit', options=[0, 1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: {
        0: 'Voiture (neuve)',
        1: 'Voiture (occasion)',
        2: 'Mobilier / Équipement',
        3: 'Radio / Télévision',
        4: 'Électroménager',
        5: 'Réparations',
        6: 'Éducation',
        7: 'Vacances / Autres'
    }[x])

st.markdown('---')

# =============================
# BOUTON DE PREDICTION
# =============================
if st.button('Évaluer le Risque Crédit', type='primary'):

    # Les colonnes doivent être dans le même ordre que X lors de l'entraînement
    input_data = np.array([[age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.markdown('---')
    st.header('Résultat de l Analyse')

    if prediction == 0:
        st.success('RISQUE FAIBLE — Ce client est susceptible de rembourser')
        st.metric(label='Score de confiance', value=f'{proba[0]*100:.1f}%')
        st.balloons()
    else:
        st.error('RISQUE ÉLEVÉ — Ce client risque de ne pas rembourser')
        st.metric(label='Probabilité de défaut', value=f'{proba[1]*100:.1f}%')

    prob_df = pd.DataFrame({
        'Catégorie': ['Bon client (remboursera)', 'Mauvais client (défaut)'],
        'Probabilité': [round(proba[0]*100, 1), round(proba[1]*100, 1)]
    })
    st.bar_chart(prob_df.set_index('Catégorie'))

# =============================
# PIED DE PAGE
# =============================
st.markdown('---')
st.caption('Projet ML Supervisé | German Credit Dataset | Modèle Lasso L1')
