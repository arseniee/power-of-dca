# app.py - Application principale CORRIGÉE et FIABLE
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy import stats
import warnings
import pdb

warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Power of DCA - Analyseur de Stratégies",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCTIONS
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

# def calculate_dca_metrics(start_date, end_date, stock_data, monthly_investment):
    """
    Calcule toutes les métriques clés pour une stratégie DCA.
    Cette version est très robuste et nettoie les données pour éviter les erreurs de calcul.
    """
    
    # 1. Filtrer, rééchantillonner et déterminer la colonne de prix
    filtered_data = stock_data.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
    monthly_data = filtered_data.resample('MS').first()
    price_column = 'Open' if 'Open' in monthly_data.columns else 'Adj Close'

    # 2. **ÉTAPE CRITIQUE** : Nettoyer les données avant tout calcul
    # Supprimer les lignes où le prix est NaN ou non positif (<= 0)
    valid_monthly_data = monthly_data[monthly_data[price_column].notna()]
    valid_monthly_data = valid_monthly_data[valid_monthly_data[price_column] > 0]

    if valid_monthly_data.empty:
        st.warning("Aucune donnée de prix mensuelle valide trouvée pour la période sélectionnée.")
        return None

    # 3. Effectuer les calculs sur les données nettoyées
    prices = valid_monthly_data[price_column]
    shares_purchased = monthly_investment / prices
    
    total_investment = monthly_investment * len(valid_monthly_data)
    total_shares = shares_purchased.sum()
        
    final_price = prices.iloc[-1]
    final_value = total_shares * final_price

    # 4. Calculer les métriques de performance et de risque
    average_purchase_price = total_investment / total_shares
    total_return_pct = (final_value - total_investment) / total_investment
    
    years_invested = len(valid_monthly_data) / 12.0
    annualized_return = ((final_value / total_investment) ** (1 / years_invested) - 1) if years_invested > 0 else 0

    portfolio_value = shares_purchased.cumsum() * prices
    
    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min()
    
    monthly_returns = portfolio_value.pct_change().dropna()
    volatility = monthly_returns.std() * np.sqrt(12) if len(monthly_returns) > 1 else 0
    sharpe_ratio = annualized_return / volatility 


    return {
                'total_investment': float(total_investment),
                'final_value': float(final_value),
                'total_shares': float(total_shares),
                'average_purchase_price': float(average_purchase_price),
                'total_return_pct': float(total_return_pct * 100),
                'annualized_return': float(annualized_return * 100),
                'volatility': float(volatility * 100),
                'max_drawdown': float(max_drawdown * 100),
                'sharpe_ratio': float(sharpe_ratio),
                'portfolio_value': portfolio_value,
                'cumulative_investment': pd.Series(np.cumsum([monthly_investment] * len(valid_monthly_data)), index=valid_monthly_data.index),
                'drawdown': drawdown
            }



def compare_strategies_rolling(investment_periods, stock1, stock2, monthly_investment, step_months):
    """
    Compare deux stratégies en utilisant une analyse par fenêtre glissante.
    """
    results1, results2, start_dates = [], [], []
    
    common_start = max(stock1.index.min(), stock2.index.min())
    common_end = min(stock1.index.max(), stock2.index.max())

    current_start = common_start
    while current_start + relativedelta(months=investment_periods) <= common_end:
        current_end = current_start + relativedelta(months=investment_periods)
        
        metrics1 = calculate_dca_metrics(current_start, current_end, stock1, monthly_investment)
        metrics2 = calculate_dca_metrics(current_start, current_end, stock2, monthly_investment)
        
        if metrics1 and metrics2:
            results1.append(metrics1)
            results2.append(metrics2)
            start_dates.append(current_start)
        
        current_start += relativedelta(months=step_months)
    
    if not results1:
        st.error("Aucune période d'investissement valide n'a pu être analysée pour la comparaison.")
        return None, None
        
    return pd.DataFrame(results1, index=start_dates), pd.DataFrame(results2, index=start_dates)

def calculate_dca_metrics(start_date, end_date, stock_data, monthly_investment):
    """
    Calcule toutes les métriques clés pour une stratégie DCA en utilisant un DataFrame
    unifié pour garantir la cohérence des données. C'est la solution la plus robuste.
    """
    try:
        # 1. Filtrer et préparer les données de base
        filtered_data = stock_data.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        monthly_data = filtered_data.resample('MS').first()
        price_column = 'Open' if 'Open' in monthly_data.columns else 'Adj Close'

        # 2. Nettoyer les données pour ne garder que les lignes avec un prix valide
        valid_monthly_data = monthly_data[monthly_data[price_column].notna() & (monthly_data[price_column] > 0)]

        if valid_monthly_data.empty:
            st.warning("Aucune donnée de prix mensuelle valide trouvée pour la période sélectionnée.")
            return None

        # 3. **SOLUTION** : Construire un DataFrame unique pour tous les calculs.
        #    Cela garantit que toutes les colonnes ont le même index et la même longueur.
        dca_df = pd.DataFrame(index=valid_monthly_data.index)
        dca_df['Price'] = valid_monthly_data[price_column]
        dca_df['SharesPurchased'] = monthly_investment / dca_df['Price']
        dca_df['CumulativeInvestment'] = np.arange(1, len(dca_df) + 1) * monthly_investment
        dca_df['CumulativeShares'] = dca_df['SharesPurchased'].cumsum()
        dca_df['PortfolioValue'] = dca_df['CumulativeShares'] * dca_df['Price']
        
        # 4. Calculer le drawdown sur ce DataFrame unifié
        peak = dca_df['PortfolioValue'].cummax()
        dca_df['Drawdown'] = (dca_df['PortfolioValue'] - peak) / peak
        
        # 5. Extraire les métriques finales (scalaires) à partir du DataFrame
        total_investment = dca_df['CumulativeInvestment'].iloc[-1]
        final_value = dca_df['PortfolioValue'].iloc[-1]
        total_shares = dca_df['CumulativeShares'].iloc[-1]
        
        if not np.isfinite(total_shares) or total_shares <= 0:
            return None

        average_purchase_price = total_investment / total_shares
        total_return_pct = (final_value - total_investment) / total_investment
        years_invested = len(dca_df) / 12.0
        annualized_return = ((final_value / total_investment) ** (1 / years_invested) - 1) if years_invested > 0 else 0
        
        max_drawdown = dca_df['Drawdown'].min()
        monthly_returns = dca_df['PortfolioValue'].pct_change().dropna()
        volatility = monthly_returns.std() * np.sqrt(12) if len(monthly_returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # 6. Renvoyer les métriques et le DataFrame unifié
        return {
            'total_investment': float(total_investment),
            'final_value': float(final_value),
            'total_shares': float(total_shares),
            'average_purchase_price': float(average_purchase_price),
            'total_return_pct': float(total_return_pct * 100),
            'annualized_return': float(annualized_return * 100),
            'volatility': float(volatility * 100),
            'max_drawdown': float(max_drawdown * 100),
            'sharpe_ratio': float(sharpe_ratio),
            'dca_df': dca_df # Le DataFrame contenant toutes les séries temporelles alignées
        }

    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul des métriques DCA : {e}")
        return None


def display_single_strategy_analysis(metrics, ticker_name):
    """
    Affiche les résultats de l'analyse en utilisant le DataFrame unifié reçu
    pour garantir que les graphiques fonctionnent toujours.
    """
    if not metrics or 'dca_df' not in metrics:
        st.error("Impossible d'afficher les résultats car les métriques n'ont pas pu être calculées.")
        return
    
    st.header(f"📈 Analyse DCA pour {ticker_name}")
    
    # Affichage des métriques scalaires (inchangé, cela fonctionnait déjà)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Investissement Total", f"${metrics['total_investment']:,.0f}")
    col1.metric("Valeur Finale", f"${metrics['final_value']:,.0f}", f"{metrics['total_return_pct']:.1f}%")
    col2.metric("Rendement Ann.", f"{metrics['annualized_return']:.1f}%")
    col2.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    col3.metric("Drawdown Max", f"{metrics['max_drawdown']:.1f}%")
    col3.metric("Ratio de Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    col4.metric("Prix Moyen", f"${metrics['average_purchase_price']:.2f}")
    col4.metric("Actions", f"{metrics['total_shares']:.2f}")

    # **SOLUTION** : Utiliser le DataFrame 'dca_df' directement pour les graphiques
    dca_df = metrics['dca_df']

    # Graphique 1: Évolution du Portefeuille
    st.subheader("📊 Évolution du Portefeuille")
    # On sélectionne les colonnes directement, Streamlit sait comment les tracer.
    st.line_chart(dca_df[['PortfolioValue', 'CumulativeInvestment']])

    # Graphique 2: Analyse du Drawdown
    st.subheader("📉 Analyse du Drawdown")
    # On sélectionne la colonne Drawdown et on la passe à st.area_chart
    st.area_chart(dca_df['Drawdown'] * 100)

def display_strategy_comparison(res1, res2, name1, name2):
    """
    Affiche la comparaison des résultats entre deux stratégies, avec des graphiques corrigés.
    """
    st.header(f"⚖️ Comparaison : {name1} vs. {name2}")
    
    if res1 is None or res2 is None or res1.empty or res2.empty:
        st.error("Les données de comparaison sont manquantes ou incomplètes.")
        return

    # Métriques
    mean_return1, mean_return2 = res1['annualized_return'].mean(), res2['annualized_return'].mean()
    win_rate1 = (res1['final_value'] > res2['final_value']).sum() / len(res1) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Rendement Ann. Moyen ({name1})", f"{mean_return1:.1f}%")
    col1.metric(f"Rendement Ann. Moyen ({name2})", f"{mean_return2:.1f}%")
    col2.metric(f"Taux de Victoire ({name1})", f"{win_rate1:.1f}%", help=f"Pourcentage des périodes où {name1} a surperformé {name2}.")
    
    # Analyse statistique
    t_stat, p_value = stats.ttest_ind(res1['annualized_return'].dropna(), res2['annualized_return'].dropna())
    col3.metric("P-value (Test T)", f"{p_value:.4f}", help="Une p-value < 0.05 suggère une différence statistiquement significative.")
    if p_value < 0.05:
        st.success("La différence de performance est statistiquement significative.")
    else:
        st.info("La différence de performance n'est pas statistiquement significative.")

    # **CORRECTION DU GRAPHIQUE** : Combiner les données pour un affichage correct
    st.subheader("📊 Distribution des Rendements Annualisés")
    df1 = pd.DataFrame({'Rendement': res1['annualized_return'], 'Stratégie': name1})
    df2 = pd.DataFrame({'Rendement': res2['annualized_return'], 'Stratégie': name2})
    combined_df = pd.concat([df1, df2])
    
    fig = px.histogram(
        combined_df, 
        x='Rendement', 
        color='Stratégie',
        marginal="box",
        barmode='overlay',
        opacity=0.7,
        title="Distribution des Rendements Annualisés sur toutes les Périodes"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# INTERFACE UTILISATEUR (UI)
# ============================================================================

ETF_LIST = {
    "SPDR S&P 500 (SPY)": "SPY", "Invesco QQQ (QQQ)": "QQQ", "Vanguard Total Stock (VTI)": "VTI",
    "iShares Russell 2000 (IWM)": "IWM", "SPDR Gold Trust (GLD)": "GLD", "iShares MSCI Em. Mkts (EEM)": "EEM",
    "iShares 20+y Treasury (TLT)": "TLT", "ARK Innovation (ARKK)": "ARKK", "Schwab Dividend (SCHD)": "SCHD",
    "Vanguard Total World (VT)": "VT", "iShares Bitcoin (IBIT)": "IBIT"
}

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
    else:
        st.subheader("Paramètres")
        stock1_name = st.selectbox("ETF #1", list(ETF_LIST.keys()), index=0)
        stock2_name = st.selectbox("ETF #2", list(ETF_LIST.keys()), index=1)
        investment_years = st.slider("Durée d'investissement (ans)", 1, 15, 5)
        investment_periods = investment_years * 12
        step_months = st.select_slider("Pas de la fenêtre (mois)", [1, 3, 6, 12], 3)
        run_button = st.button("🚀 Lancer la Comparaison", type="primary", use_container_width=True)

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
                    metrics = calculate_dca_metrics(start_date, end_date, stock_data, monthly_investment)
                    display_single_strategy_analysis(metrics, stock_name)
        else:
            if stock1_name == stock2_name:
                st.warning("Veuillez sélectionner deux ETF différents.")
            else:
                stock1_data = get_stock_data(ETF_LIST[stock1_name])
                stock2_data = get_stock_data(ETF_LIST[stock2_name])
                if stock1_data is not None and stock2_data is not None:
                    res1, res2 = compare_strategies_rolling(investment_periods, stock1_data, stock2_data, monthly_investment, step_months)
                    display_strategy_comparison(res1, res2, stock1_name, stock2_name)

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Ceci n'est pas un conseil financier. Faites toujours vos propres recherches.</div>", unsafe_allow_html=True)