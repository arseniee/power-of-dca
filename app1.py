# app.py

import streamlit as st
import yfinance as yf
from datetime import date

# Importer les fonctions depuis les modules utilitaires
from utils_single import calculate_dca_metrics, display_single_strategy_analysis
from utils_comparison import compare_strategies_rolling, display_strategy_comparison

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Power of DCA - Analyseur de Stratégies",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600)  # Mise en cache des données pour 1 heure
def get_stock_data(ticker, start_date='2000-01-01', end_date=None):
    """
    Récupère les données historiques d'un ticker depuis Yahoo Finance avec une gestion d'erreurs robuste.
    """
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            st.error(f"Aucune donnée trouvée pour le ticker {ticker} dans la plage de dates spécifiée.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données pour {ticker}: {str(e)}")
        return None

# ============================================================================
# INTERFACE UTILISATEUR (UI)
# ============================================================================

# Constantes pour l'UI
ETF_LIST = {
    "SPDR S&P 500 (SPY)": "SPY", "Invesco QQQ (QQQ)": "QQQ", "Invesco QQQ 3x Daily L": "QQQ3.MI","Vanguard Total Stock (VTI)": "VTI",
    "iShares Russell 2000 (IWM)": "IWM", "SPDR Gold Trust (GLD)": "GLD", "iShares MSCI Em. Mkts (EEM)": "EEM",
    "iShares 20+y Treasury (TLT)": "TLT", "ARK Innovation (ARKK)": "ARKK", "Schwab Dividend (SCHD)": "SCHD",
    "Vanguard Total World (VT)": "VT", "iShares Bitcoin (IBIT)": "IBIT"
}

# Barre latérale de configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    analysis_type = st.radio("Type d'Analyse", ["Analyse Simple", "Comparaison"], horizontal=True)
    monthly_investment = st.number_input("Investissement Mensuel ($)", 50, 10000, 500, 50)
    
    if analysis_type == "Analyse Simple":
        st.subheader("Paramètres")
        stock_name = st.selectbox("Choisissez un ETF", list(ETF_LIST.keys()))
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Début", date(2020, 1, 1))
        end_date = col2.date_input("Fin", date.today())
        run_button = st.button("🚀 Lancer l'Analyse", type="primary", use_container_width=True)
    else: # Comparaison
        st.subheader("Paramètres")
        stock1_name = st.selectbox("ETF #1", list(ETF_LIST.keys()), index=0)
        stock2_name = st.selectbox("ETF #2", list(ETF_LIST.keys()), index=1)
        investment_years = st.slider("Durée d'investissement (ans)", 1, 15, 5)
        investment_periods = investment_years * 12
        step_months = st.select_slider("Pas de la fenêtre (mois)", [1, 3, 6, 12], 3)
        run_button = st.button("🚀 Lancer la Comparaison", type="primary", use_container_width=True)

# Contenu principal de la page
st.title("Power of DCA")
st.markdown("Un outil pour analyser et comparer des stratégies d'investissement programmé (Dollar-Cost Averaging).")

if run_button:
    with st.spinner("Analyse en cours..."):
        if analysis_type == "Analyse Simple":
            if start_date >= end_date:
                st.error("La date de début doit être antérieure à la date de fin.")
            else:
                stock_data = get_stock_data(ETF_LIST[stock_name], start_date, end_date)
                if stock_data is not None:
                    # Appel des fonctions du module utils_single
                    metrics = calculate_dca_metrics(start_date, end_date, stock_data, monthly_investment)
                    display_single_strategy_analysis(metrics, stock_name)
        else: # Comparaison
            if stock1_name == stock2_name:
                st.warning("Veuillez sélectionner deux ETF différents.")
            else:
                stock1_data = get_stock_data(ETF_LIST[stock1_name])
                stock2_data = get_stock_data(ETF_LIST[stock2_name])
                if stock1_data is not None and stock2_data is not None:
                    # Appel des fonctions du module utils_comparison
                    res1, res2 = compare_strategies_rolling(investment_periods, stock1_data, stock2_data, monthly_investment, step_months)
                    display_strategy_comparison(res1, res2, stock1_name, stock2_name)

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Ceci n'est pas un conseil financier. Faites toujours vos propres recherches.</div>", unsafe_allow_html=True)