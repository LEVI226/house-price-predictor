# =============================================================================
# 🏡 APPLICATION STREAMLIT - PRÉDICTION PRIX DES MAISONS
# Yannick OUEDRAOGO - Interface Web pour Modèle ML
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 🎨 CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="🏡 Prédicteur Prix Maisons",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .conversion-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔧 FONCTIONS UTILITAIRES
# =============================================================================

@st.cache_data
def load_model_and_info():
    """Charger le modèle et les informations."""
    try:
        # Charger le modèle
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Charger les informations
        with open('feature_info.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        
        return model, feature_info
    except FileNotFoundError:
        st.error("❌ Fichiers modèle non trouvés ! Assurez-vous que 'xgb_model.pkl' et 'feature_info.pkl' sont dans le même dossier.")
        return None, None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None, None

def sqft_to_m2(sqft):
    """Convertir pieds carrés en mètres carrés."""
    return sqft * 0.092903

def m2_to_sqft(m2):
    """Convertir mètres carrés en pieds carrés."""
    return m2 / 0.092903

def usd_to_eur(usd, rate=0.85):
    """Convertir USD en EUR (taux approximatif)."""
    return usd * rate

def predict_price(model, features):
    """Faire une prédiction de prix."""
    try:
        prediction = model.predict([features])[0]
        return max(0, prediction)  # Éviter les prix négatifs
    except Exception as e:
        st.error(f"Erreur de prédiction: {str(e)}")
        return 0

def create_feature_importance_chart(feature_info):
    """Créer un graphique d'importance des features."""
    if feature_info.get('feature_importance'):
        df_importance = pd.DataFrame(feature_info['feature_importance'])
        
        fig = px.bar(
            df_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title="🔍 Importance des Caractéristiques",
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    return None

def create_prediction_confidence_chart(prediction, feature_info):
    """Créer un graphique de confiance de prédiction."""
    rmse = feature_info.get('rmse_score', 26240)
    
    # Intervalle de confiance approximatif
    lower_bound = prediction - rmse
    upper_bound = prediction + rmse
    
    fig = go.Figure()
    
    # Barre de prédiction
    fig.add_trace(go.Bar(
        x=['Prédiction'],
        y=[prediction],
        name='Prix prédit',
        marker_color='#1f77b4',
        text=[f'${prediction:,.0f}'],
        textposition='auto'
    ))
    
    # Intervalle de confiance
    fig.add_trace(go.Scatter(
        x=['Prédiction', 'Prédiction'],
        y=[lower_bound, upper_bound],
        mode='markers+lines',
        name='Intervalle de confiance',
        marker=dict(size=10, color='red'),
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="🎯 Prédiction avec Intervalle de Confiance",
        yaxis_title="Prix ($)",
        showlegend=True,
        height=400
    )
    
    return fig

# =============================================================================
# 🎯 INTERFACE PRINCIPALE
# =============================================================================

def main():
    # Titre principal
    st.markdown("# 🏡 Prédicteur de Prix de Maisons")
    st.markdown("### 🤖 Modèle ML XGBoost Optimisé par Yannick OUEDRAOGO")
    st.markdown("---")
    
    # Charger le modèle et les informations
    model, feature_info = load_model_and_info()
    
    if model is None or feature_info is None:
        st.stop()
    
    # Informations sur le modèle
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Précision R²", 
            f"{feature_info['model_stats']['test_r2']:.1%}",
            help="Pourcentage de variance expliquée par le modèle"
        )
    
    with col2:
        st.metric(
            "📊 Erreur RMSE", 
            f"${feature_info['rmse_score']:,.0f}",
            help="Erreur quadratique moyenne sur les données de test"
        )
    
    with col3:
        st.metric(
            "🏠 Prix Moyen", 
            f"${feature_info['model_stats']['mean_price']:,.0f}",
            help="Prix moyen des maisons dans le dataset"
        )
    
    with col4:
        st.metric(
            "📈 Échantillons", 
            "1,456 maisons",
            help="Nombre de maisons utilisées pour l'entraînement"
        )
    
    st.markdown("---")
    
    # Interface utilisateur divisée en deux colonnes
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("## 📝 Caractéristiques de la Maison")
        
        # Information sur les unités
        st.markdown("""
        <div class="conversion-info">
        <h4>ℹ️ Information sur les Unités</h4>
        <p>Les surfaces sont demandées en <strong>mètres carrés (m²)</strong> pour votre confort. 
        Le modèle effectue automatiquement la conversion en pieds carrés pour la prédiction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Créer deux colonnes pour les inputs
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Surface habitable (en m²)
            gr_liv_area_m2 = st.number_input(
                "🏠 Surface Habitable (m²)",
                min_value=50.0,
                max_value=500.0,
                value=150.0,
                step=5.0,
                help="Surface totale habitable de la maison"
            )
            
            # Surface sous-sol (en m²)
            total_bsmt_sf_m2 = st.number_input(
                "🔧 Surface Sous-sol (m²)",
                min_value=0.0,
                max_value=300.0,
                value=100.0,
                step=5.0,
                help="Surface totale du sous-sol"
            )
            
            # Qualité générale
            overall_qual = st.selectbox(
                "⭐ Qualité Générale",
                options=list(range(1, 11)),
                index=5,
                help="Qualité générale de la maison (1=Très mauvaise, 10=Excellente)"
            )
            
            # Nombre de places de garage
            garage_cars = st.selectbox(
                "🚗 Places de Garage",
                options=[0, 1, 2, 3, 4],
                index=2,
                help="Nombre de voitures que peut accueillir le garage"
            )
            
            # Surface garage (en m²)
            garage_area_m2 = st.number_input(
                "🏠 Surface Garage (m²)",
                min_value=0.0,
                max_value=150.0,
                value=40.0,
                step=5.0,
                help="Surface totale du garage"
            )
        
        with col_right:
            # Année de construction
            year_built = st.number_input(
                "📅 Année de Construction",
                min_value=1850,
                max_value=datetime.now().year,
                value=2000,
                step=1,
                help="Année de construction de la maison"
            )
            
            # Nombre de salles de bain complètes
            full_bath = st.selectbox(
                "🛁 Salles de Bain Complètes",
                options=[0, 1, 2, 3, 4],
                index=2,
                help="Nombre de salles de bain avec baignoire/douche, lavabo et toilettes"
            )
            
            # Nombre total de pièces
            tot_rms_abv_grd = st.number_input(
                "🏠 Nombre de Pièces",
                min_value=3,
                max_value=15,
                value=7,
                step=1,
                help="Nombre total de pièces au-dessus du niveau du sol"
            )
            
            # Nombre de cheminées
            fireplaces = st.selectbox(
                "🔥 Cheminées",
                options=[0, 1, 2, 3],
                index=1,
                help="Nombre de cheminées dans la maison"
            )
        
        # Bouton de prédiction
        st.markdown("---")
        predict_button = st.button("🔮 Prédire le Prix", type="primary", use_container_width=True)
    
    with col_output:
        st.markdown("## 🎯 Résultat de la Prédiction")
        
        if predict_button:
            # Convertir les surfaces en pieds carrés pour le modèle
            gr_liv_area_sqft = m2_to_sqft(gr_liv_area_m2)
            total_bsmt_sf_sqft = m2_to_sqft(total_bsmt_sf_m2)
            garage_area_sqft = m2_to_sqft(garage_area_m2)
            
            # Préparer les features dans l'ordre attendu par le modèle
            features = [
                gr_liv_area_sqft,      # GrLivArea
                total_bsmt_sf_sqft,    # TotalBsmtSF
                overall_qual,          # OverallQual
                garage_cars,           # GarageCars
                garage_area_sqft,      # GarageArea
                year_built,            # YearBuilt
                full_bath,             # FullBath
                tot_rms_abv_grd,       # TotRmsAbvGrd
                fireplaces             # Fireplaces
            ]
            
            # Faire la prédiction
            prediction_usd = predict_price(model, features)
            prediction_eur = usd_to_eur(prediction_usd)
            
            # Afficher le résultat principal
            st.markdown(f"""
            <div class="prediction-card">
                <h2>💰 Prix Prédit</h2>
                <h1>${prediction_usd:,.0f}</h1>
                <h3>≈ {prediction_eur:,.0f} €</h3>
                <p>Intervalle de confiance: ±${feature_info['rmse_score']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métriques détaillées
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                st.metric(
                    "💵 Prix Minimum", 
                    f"${prediction_usd - feature_info['rmse_score']:,.0f}",
                    help="Estimation basse (prix - RMSE)"
                )
            
            with col_met2:
                st.metric(
                    "💎 Prix Maximum", 
                    f"${prediction_usd + feature_info['rmse_score']:,.0f}",
                    help="Estimation haute (prix + RMSE)"
                )
            
            with col_met3:
                price_per_m2 = prediction_usd / (gr_liv_area_m2 + total_bsmt_sf_m2 + garage_area_m2)
                st.metric(
                    "📏 Prix/m²", 
                    f"${price_per_m2:,.0f}",
                    help="Prix par mètre carré total"
                )
            
            # Graphique de confiance
            confidence_chart = create_prediction_confidence_chart(prediction_usd, feature_info)
            if confidence_chart:
                st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Résumé des caractéristiques
            st.markdown("### 📋 Résumé des Caractéristiques")
            
            summary_data = {
                "Caractéristique": [
                    "Surface Habitable", "Surface Sous-sol", "Surface Garage",
                    "Qualité Générale", "Places Garage", "Année Construction",
                    "Salles de Bain", "Nombre Pièces", "Cheminées"
                ],
                "Valeur": [
                    f"{gr_liv_area_m2:.0f} m² ({gr_liv_area_sqft:.0f} sq ft)",
                    f"{total_bsmt_sf_m2:.0f} m² ({total_bsmt_sf_sqft:.0f} sq ft)",
                    f"{garage_area_m2:.0f} m² ({garage_area_sqft:.0f} sq ft)",
                    f"{overall_qual}/10",
                    f"{garage_cars} voiture(s)",
                    f"{year_built}",
                    f"{full_bath}",
                    f"{tot_rms_abv_grd}",
                    f"{fireplaces}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # Section d'analyse avancée
    st.markdown("---")
    st.markdown("## 📊 Analyse Avancée")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Importance Features", "📈 Statistiques Modèle", "ℹ️ À propos"])
    
    with tab1:
        st.markdown("### 🔍 Importance des Caractéristiques")
        
        # Graphique d'importance
        importance_chart = create_feature_importance_chart(feature_info)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
        
        st.markdown("""
        **Interprétation:**
        - **OverallQual** est de loin le facteur le plus important (~58%)
        - **GrLivArea** (surface habitable) vient en second (~18%)
        - **TotalBsmtSF** (surface sous-sol) contribue également (~9%)
        - Les autres caractéristiques ont un impact plus modéré
        """)
    
    with tab2:
        st.markdown("### 📈 Statistiques du Modèle")
        
        stats = feature_info['model_stats']
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("**📊 Performance Train:**")
            st.write(f"• R² Score: {stats['train_r2']:.4f}")
            st.write(f"• RMSE: ${stats['train_rmse']:,.2f}")
            
            st.markdown("**🎯 Performance Test:**")
            st.write(f"• R² Score: {stats['test_r2']:.4f}")
            st.write(f"• RMSE: ${stats['test_rmse']:,.2f}")
        
        with col_stat2:
            st.markdown("**📈 Statistiques Prix:**")
            st.write(f"• Prix Moyen: ${stats['mean_price']:,.2f}")
            st.write(f"• Écart-type: ${stats['std_price']:,.2f}")
            
            overfitting = abs(stats['train_rmse'] - stats['test_rmse'])
            st.markdown("**🔍 Analyse:**")
            st.write(f"• Différence Train/Test: ${overfitting:,.2f}")
            st.write(f"• Overfitting: {'Minimal' if overfitting < 5000 else 'Modéré'}")
        
        if 'best_params' in feature_info:
            st.markdown("**⚙️ Hyperparamètres Optimaux:**")
            params_df = pd.DataFrame(list(feature_info['best_params'].items()), 
                                   columns=['Paramètre', 'Valeur'])
            st.dataframe(params_df, hide_index=True, use_container_width=True)
    
    with tab3:
        st.markdown("### ℹ️ À propos de cette Application")
        
        st.markdown("""
        **🏡 Prédicteur de Prix de Maisons**
        
        Cette application utilise un modèle XGBoost optimisé pour prédire le prix des maisons 
        basé sur leurs caractéristiques principales.
        
        **📊 Données:**
        - Dataset: Ames Housing (Iowa, USA)
        - Échantillons: 1,456 maisons
        - Features: 9 caractéristiques principales
        
        **🤖 Modèle:**
        - Algorithme: XGBoost (Gradient Boosting optimisé)
        - Précision: 86.9% (R² Score)
        - Erreur moyenne: ±$26,240
        
        **👨‍💻 Développeur:**
        - Yannick OUEDRAOGO
        - Projet End-to-End Machine Learning
        
        **🔧 Technologies:**
        - Python, Scikit-learn, XGBoost
        - Streamlit, Plotly
        - Pandas, NumPy
        
        **📝 Note sur les Conversions:**
        Les surfaces sont saisies en mètres carrés pour votre confort, mais le modèle 
        fonctionne en pieds carrés (unité originale du dataset américain).
        
        **⚠️ Limitations:**
        - Modèle entraîné sur des données de l'Iowa (USA)
        - Prix en dollars américains
        - Précision variable selon les caractéristiques de la maison
        """)

# =============================================================================
# 🚀 POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()

# =============================================================================
# 📋 INSTRUCTIONS D'UTILISATION
# =============================================================================

"""
📝 INSTRUCTIONS POUR LANCER L'APPLICATION:

1. Assurez-vous d'avoir les fichiers suivants dans le même dossier:
   - ce fichier Python (ex: app.py)
   - xgb_model.pkl (modèle entraîné)
   - feature_info.pkl (métadonnées)

2. Installez les dépendances:
   pip install streamlit pandas numpy plotly matplotlib seaborn

3. Lancez l'application:
   streamlit run app.py

4. L'application s'ouvrira dans votre navigateur à l'adresse:
   http://localhost:8501

🎯 FONCTIONNALITÉS:
- Interface intuitive en mètres carrés
- Prédiction en temps réel
- Visualisations interactives
- Intervalle de confiance
- Analyse des caractéristiques importantes
- Statistiques détaillées du modèle

💡 CONSEILS D'UTILISATION:
- Utilisez des valeurs réalistes pour de meilleures prédictions
- Consultez l'onglet "Importance Features" pour comprendre l'impact
- L'intervalle de confiance donne une estimation de l'incertitude
"""
