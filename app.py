import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="🏡 Prédicteur de Prix de Maisons",
    page_icon="🏡",
    layout="wide"
)

# Titre principal
st.title("🏡 Prédicteur de Prix de Maisons")
st.markdown("### Obtenez une estimation du prix de votre maison en quelques clics !")

# Sidebar pour les inputs
st.sidebar.header("🔧 Caractéristiques de la maison")

# Création des sliders avec les vraies plages de données
gr_liv_area = st.sidebar.slider(
    "Surface habitable (sq ft)", 
    min_value=ranges['GrLivArea']['min'], 
    max_value=ranges['GrLivArea']['max'], 
    value=int((ranges['GrLivArea']['min'] + ranges['GrLivArea']['max']) / 2),
    help="Surface totale de la zone habitable"
)

total_bsmt_sf = st.sidebar.slider(
    "Surface sous-sol (sq ft)", 
    min_value=ranges['TotalBsmtSF']['min'], 
    max_value=ranges['TotalBsmtSF']['max'], 
    value=int((ranges['TotalBsmtSF']['min'] + ranges['TotalBsmtSF']['max']) / 2),
    help="Surface totale du sous-sol"
)

overall_qual = st.sidebar.selectbox(
    "Qualité générale", 
    options=list(range(ranges['OverallQual']['min'], ranges['OverallQual']['max'] + 1)),
    index=2,
    help="Note de 1 (très pauvre) à 10 (excellent)"
)

garage_cars = st.sidebar.selectbox(
    "Nombre de places de garage",
    options=list(range(ranges['GarageCars']['min'], ranges['GarageCars']['max'] + 1)),
    index=1,
    help="Nombre de voitures que peut accueillir le garage"
)

garage_area = st.sidebar.slider(
    "Surface garage (sq ft)",
    min_value=ranges['GarageArea']['min'],
    max_value=ranges['GarageArea']['max'],
    value=int((ranges['GarageArea']['min'] + ranges['GarageArea']['max']) / 2),
    help="Surface du garage en pieds carrés"
)

# Fonction pour charger votre vrai modèle
@st.cache_data
def load_model_and_info():
    try:
        model = pickle.load(open('xgb_model.pkl', 'rb'))
        feature_info = pickle.load(open('feature_info.pkl', 'rb'))
        return model, feature_info
    except FileNotFoundError:
        st.error("❌ Fichiers modèle non trouvés ! Assurez-vous d'avoir xgb_model.pkl et feature_info.pkl dans le dossier.")
        st.stop()

# Charger le modèle et les informations
model, feature_info = load_model_and_info()

# Utiliser les vraies plages de valeurs
ranges = feature_info['feature_ranges']

# Prédiction
if st.sidebar.button("🔮 Prédire le prix", type="primary"):
    # Préparer les données
    input_data = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'TotalBsmtSF': [total_bsmt_sf],
        'OverallQual': [overall_qual],
        'GarageCars': [garage_cars],
        'GarageArea': [garage_area]
    })
    
    # Faire la prédiction
    prediction = model.predict(input_data)[0]
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### 💰 Prix estimé : ${prediction:,.0f}")
        
        # Fourchette de prix (±10%)
        lower_bound = prediction * 0.9
        upper_bound = prediction * 1.1
        st.info(f"**Fourchette:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
        
    with col2:
        # Graphique de comparaison
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['Votre maison', 'Prix moyen\ndu marché']
        values = [prediction, 200000]  # Prix moyen fictif
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Prix ($)')
        ax.set_title('Comparaison avec le marché')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                   f'${value:,.0f}', ha='center', va='bottom')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

# Section informative
st.markdown("---")
st.markdown("## 📊 À propos du modèle")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 Précision du modèle", f"RMSE: {feature_info['rmse_score']:,.0f}", "XGBoost entraîné")

with col2:
    st.metric("📈 Algorithme utilisé", "XGBoost", "Gradient Boosting")

with col3:
    st.metric("🏠 Données d'entraînement", "1,460 maisons", "Dataset Kaggle")

# Explications des features
with st.expander("ℹ️ Explication des caractéristiques"):
    st.markdown("""
    **Surface habitable (GrLivArea):** Surface totale au-dessus du niveau du sol
    
    **Surface sous-sol (TotalBsmtSF):** Surface totale du sous-sol
    
    **Qualité générale (OverallQual):** Évaluation globale du matériau et de la finition
    - 1-3: Pauvre à moyen-faible
    - 4-6: Moyen à moyen-élevé  
    - 7-9: Bon à excellent
    - 10: Très excellent
    
    **Places de garage:** Capacité du garage en nombre de voitures
    
    **Surface garage:** Superficie du garage en pieds carrés
    """)

# Footer
st.markdown("---")
st.markdown("*Développé avec ❤️ par Yannick OUEDRAOGO - Projet Machine Learning*")