# utils_single.py

import streamlit as st
import pandas as pd
import numpy as np

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

        # 3. Construire un DataFrame unique pour tous les calculs.
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
        volatility = monthly_returns.std() if len(monthly_returns) > 1 else 0
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
            'dca_df': dca_df 
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
    
    # Affichage des métriques scalaires
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Investissement Total", f"${metrics['total_investment']:,.0f}")
    col1.metric("Valeur Finale", f"${metrics['final_value']:,.0f}", f"{metrics['total_return_pct']:.1f}%")
    col2.metric("Rendement Ann.", f"{metrics['annualized_return']:.1f}%")
    col2.metric("Volatilité", f"{metrics['volatility']:.1f}%")
    col3.metric("Drawdown Max", f"{metrics['max_drawdown']:.1f}%")
    col3.metric("Ratio de Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    col4.metric("Prix Moyen", f"${metrics['average_purchase_price']:.2f}")
    col4.metric("Actions", f"{metrics['total_shares']:.2f}")

    # Utiliser le DataFrame 'dca_df' directement pour les graphiques
    dca_df = metrics['dca_df']

    # Graphique 1: Évolution du Portefeuille
    st.subheader("📊 Évolution du Portefeuille")
    st.line_chart(dca_df[['PortfolioValue', 'CumulativeInvestment']])

    # Graphique 2: Analyse du Drawdown
    st.subheader("📉 Analyse du Drawdown")
    st.area_chart(dca_df['Drawdown'] * 100)