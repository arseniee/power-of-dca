import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime 
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import FuncFormatter
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats





# Functions single stock
def test_strat_graphs(startDate, endDate, stock_data, monthlyInvestment):
    # Filter data for the specified date range
    data_range = stock_data.loc[startDate:endDate].resample('MS').first()
    
    # Calculate total investment and shares
    total_investment = monthlyInvestment * len(data_range)
    shares_purchased = monthlyInvestment / data_range['Open']
    total_shares = shares_purchased.sum()
    average_purchase_price = total_investment / total_shares
    
    # Calculate portfolio metrics
    cumulative_investment = np.cumsum([monthlyInvestment] * len(data_range))
    portfolio_value = shares_purchased.cumsum() * data_range['Open'].to_numpy()

    cumulative_returns = (portfolio_value - cumulative_investment[:,np.newaxis]) / cumulative_investment[:,np.newaxis]
    monthly_returns = portfolio_value.pct_change().dropna()
    
    # Calculate performance statistics
    final_value = portfolio_value.iloc[-1]
    total_return_pct = (final_value - total_investment) / total_investment * 100
    annualized_return = (final_value / total_investment) ** (12/len(data_range)) - 1
    
    # Calculate drawdown
    peak = portfolio_value.cummax()
    drawdown = ((portfolio_value - peak) / peak).squeeze()
    
    # --- Create figure ---
    fig = plt.figure(figsize=(16, 14))

    # Portfolio Value vs Investment
    ax1 = plt.subplot(3, 2, 1)
    portfolio_value.plot(ax=ax1, color='royalblue', label='Portfolio Value')
    ax1.plot(cumulative_investment, '--', color='darkorange', label='Cumulative Investment')
    ax1.set_title('Portfolio Value vs Investment', fontsize=12)
    ax1.set_xlabel('')
    ax1.legend()
    ax1.grid(True)

    # Cumulative Returns
    ax2 = plt.subplot(3, 2, 2)
    cumulative_returns.plot(ax=ax2, color='forestgreen')
    ax2.set_title('Cumulative Returns (%)', fontsize=12)
    ax2.set_ylabel('Return Percentage')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True)

    # Drawdown
    ax3 = plt.subplot(3, 2, 3)
    drawdown.plot(ax=ax3, color='crimson')
    ax3.fill_between(drawdown.index, drawdown.values, color='crimson', alpha=0.3)
    ax3.set_title('Portfolio Drawdown', fontsize=12)
    ax3.set_ylabel('Drawdown')
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.grid(True)

    # Shares Accumulation
    ax5 = plt.subplot(3, 2, 4)
    shares_purchased.cumsum().plot(ax=ax5, color='purple')
    ax5.set_title('Shares Accumulation Over Time', fontsize=12)
    ax5.set_xlabel('')
    ax5.set_ylabel('Total Shares')
    ax5.grid(True)

    # Stock Price with Investment Points
    ax6 = plt.subplot(3, 2, 5)
    stock_data['Close'].loc[startDate:endDate].plot(ax=ax6, color='gray', alpha=0.5, label='Daily Price')
    data_range['Open'].plot(ax=ax6, style='o', markersize=5, color='blue', label='Investment Points')
    ax6.set_title('Stock Price with Investment Points', fontsize=12)
    ax6.set_ylabel('Price ($)')
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()

    # --- Display in Streamlit ---
    st.pyplot(fig)
    
    
    results =  {
        'total_investment': total_investment,
        'final_value': final_value.values[0],
        'total_shares': total_shares.values[0],
        'average_purchase_price': average_purchase_price.values[0],
        'total_return_pct': total_return_pct.values[0],
        'annualized_return': annualized_return.values[0],
        'monthly_returns': monthly_returns,
        'drawdown': drawdown
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Investment", f"${results['total_investment']:,.2f}")
        st.metric("Final Value", f"${results['final_value']:,.2f}")

    with col2:
        st.metric("Average Purchase Price", f"${results['average_purchase_price']:,.2f}")
        st.metric("Total Shares", f"{results['total_shares']:.2f}")


    with col3:
        st.metric("Total Return (%)", f"{results['total_return_pct']:.2f}%")
        st.metric("Annualized Return", f"{results['annualized_return']*100:.2f}%")

    
    








# Functions Copmare
def get_stock_data(ticker, start_date='1900-01-01', end_date='2025-07-31'):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def test_strat(startDate, endDate, stock_data, monthlyInvestment):
    # Filter data for the specified date range
    data_range = stock_data.loc[startDate:endDate].resample('MS').first()
    
    # Calculate total investment and shares
    total_investment = monthlyInvestment * len(data_range)
    shares_purchased = monthlyInvestment / data_range['Open']
    total_shares = shares_purchased.sum()
    average_purchase_price = total_investment / total_shares
    
    # Calculate portfolio metrics
    cumulative_investment = np.cumsum([monthlyInvestment] * len(data_range))
    portfolio_value = shares_purchased.cumsum() * data_range['Open'].to_numpy()

    cumulative_returns = (portfolio_value - cumulative_investment[:,np.newaxis]) / cumulative_investment[:,np.newaxis]
    monthly_returns = portfolio_value.pct_change().dropna()
    
    # Calculate performance statistics
    final_value = portfolio_value.iloc[-1]
    total_return_pct = (final_value - total_investment) / total_investment * 100
    annualized_return = (final_value / total_investment) ** (12/len(data_range)) - 1
    
    # Calculate drawdown
    peak = portfolio_value.cummax()
    drawdown = ((portfolio_value - peak) / peak).squeeze()
    
    
    return {
        'total_investment': total_investment,
        'final_value': final_value.values[0],
        'total_shares': total_shares.values[0],
        'average_purchase_price': average_purchase_price.values[0],
        'total_return_pct': total_return_pct.values[0],
        'annualized_return': annualized_return.values[0],
        'monthly_returns': monthly_returns.to_numpy().squeeze(),
        'drawdown': drawdown.to_numpy().squeeze()
    }



def compare_strats(investmentPeriods, stock1, stock2, monthlyInvestment):
    res1 = pd.DataFrame([], columns= ['total_investment','final_value','total_shares','average_purchase_price','total_return_pct','annualized_return','monthly_returns','drawdown'])
    res2 = pd.DataFrame([], columns= ['total_investment','final_value','total_shares','average_purchase_price','total_return_pct','annualized_return','monthly_returns','drawdown'])

    startDate = max(stock1.index.min(), stock2.index.min())
    endDate = pd.to_datetime(startDate) + relativedelta(months=investmentPeriods)
    while endDate < pd.to_datetime('2025-07-31'):
        try:
            res1.loc[startDate] = test_strat(startDate, endDate, stock1, monthlyInvestment)
            res2.loc[startDate] = test_strat(startDate, endDate, stock2, monthlyInvestment)
            startDate = startDate + relativedelta(months=6)
            endDate = endDate + relativedelta(months=6)
        except:
            startDate = startDate + relativedelta(months=6)
            endDate = endDate + relativedelta(months=6)
            continue

    return res1, res2



def compare_investment_strategies_streamlit(res1, res2, strategy1_name="Stratégie 1", strategy2_name="Stratégie 2"):
    """
    Compare et visualise deux stratégies d'investissement dans Streamlit
    
    Parameters:
    res1, res2: DataFrames contenant les résultats des stratégies
    strategy1_name, strategy2_name: Noms des stratégies pour les graphiques
    """
    
    # Préparation des données
    def clean_data(df):
        """Nettoie les données et gère les valeurs manquantes"""
        df_clean = df.copy()
        for col in ['total_return_pct', 'annualized_return']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean.dropna()
    
    res1_clean = clean_data(res1)
    res2_clean = clean_data(res2)
    
    # Configuration de la page
    st.title("🔍 Analyse Comparative des Stratégies d'Investissement")
    st.markdown("---")
    
    # Sidebar avec contrôles
    with st.sidebar:
        st.header("⚙️ Configuration")
        show_stats = st.checkbox("Afficher les statistiques détaillées", value=True)
        show_tests = st.checkbox("Afficher les tests statistiques", value=True)
        chart_height = st.slider("Hauteur des graphiques", 400, 800, 500)
    
    # Métriques principales en haut
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rendement Moyen", 
            f"{res1_clean['total_return_pct'].mean():.1f}%",
            f"{res1_clean['total_return_pct'].mean() - res2_clean['total_return_pct'].mean():.1f}%",
            help=f"Rendement moyen de {strategy1_name}"
        )
    
    with col2:
        st.metric(
            "Volatilité", 
            f"{res1_clean['total_return_pct'].std():.1f}%",
            f"{res1_clean['total_return_pct'].std() - res2_clean['total_return_pct'].std():.1f}%",
            help=f"Écart-type des rendements de {strategy1_name}"
        )
    
    with col3:
        wins_strategy1 = (res1_clean['total_return_pct'] > res2_clean['total_return_pct']).sum()
        win_rate_1 = wins_strategy1 / len(res1_clean) * 100
        st.metric(
            "Taux de Victoire", 
            f"{win_rate_1:.1f}%",
            help=f"Pourcentage de fois où {strategy1_name} surperforme"
        )
    
    with col4:
        sharpe_1 = res1_clean['total_return_pct'].mean() / res1_clean['total_return_pct'].std()
        sharpe_2 = res2_clean['total_return_pct'].mean() / res2_clean['total_return_pct'].std()
        st.metric(
            "Ratio Sharpe", 
            f"{sharpe_1:.2f}",
            f"{sharpe_1 - sharpe_2:.2f}",
            help="Ratio rendement/risque approximatif"
        )
    
    st.markdown("---")
    
    # Onglets pour organiser les analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "📊 Distributions", "🔗 Corrélations", "⚖️ Comparaisons"])
    
    with tab1:
        st.subheader("Évolution de la Performance")
        
        # Graphique 1: Évolution des rendements totaux
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=res1_clean.index,
            y=res1_clean['total_return_pct'],
            mode='lines+markers',
            name=strategy1_name,
            line=dict(width=3),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Rendement: %{y:.1f}%<extra></extra>'
        ))
        fig1.add_trace(go.Scatter(
            x=res2_clean.index,
            y=res2_clean['total_return_pct'],
            mode='lines+markers',
            name=strategy2_name,
            line=dict(width=3),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Rendement: %{y:.1f}%<extra></extra>'
        ))
        fig1.update_layout(
            title="Évolution des Rendements Totaux",
            xaxis_title="Date de Début d'Investissement",
            yaxis_title="Rendement Total (%)",
            height=chart_height,
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Graphique 2: Rendements annualisés
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=res1_clean.index,
            y=res1_clean['annualized_return'] * 100,
            mode='lines+markers',
            name=strategy1_name,
            line=dict(width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=res2_clean.index,
            y=res2_clean['annualized_return'] * 100,
            mode='lines+markers',
            name=strategy2_name,
            line=dict(width=3)
        ))
        fig2.update_layout(
            title="Évolution des Rendements Annualisés",
            xaxis_title="Date de Début d'Investissement",
            yaxis_title="Rendement Annualisé (%)",
            height=chart_height
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Graphique 3: Différence de performance
        cumulative_diff = res1_clean['total_return_pct'] - res2_clean['total_return_pct']
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=res1_clean.index,
            y=cumulative_diff,
            mode='lines+markers',
            name=f'Différence ({strategy1_name} - {strategy2_name})',
            line=dict(width=3, color='green'),
            fill='tonexty' if cumulative_diff.mean() > 0 else 'tozeroy',
            fillcolor='rgba(0,255,0,0.1)' if cumulative_diff.mean() > 0 else 'rgba(255,0,0,0.1)'
        ))
        fig3.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Égalité")
        fig3.update_layout(
            title=f"Surperformance de {strategy1_name} vs {strategy2_name}",
            xaxis_title="Date de Début d'Investissement",
            yaxis_title="Différence de Rendement (%)",
            height=chart_height
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.subheader("Analyse des Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogrammes superposés
            fig4 = go.Figure()
            fig4.add_trace(go.Histogram(
                x=res1_clean['total_return_pct'],
                name=strategy1_name,
                opacity=0.7,
                nbinsx=15
            ))
            fig4.add_trace(go.Histogram(
                x=res2_clean['total_return_pct'],
                name=strategy2_name,
                opacity=0.7,
                nbinsx=15
            ))
            fig4.update_layout(
                title="Distribution des Rendements Totaux",
                xaxis_title="Rendement Total (%)",
                yaxis_title="Fréquence",
                barmode='overlay',
                height=chart_height//1.2
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Box plots
            fig5 = go.Figure()
            fig5.add_trace(go.Box(
                y=res1_clean['total_return_pct'],
                name=strategy1_name,
                boxpoints='outliers'
            ))
            fig5.add_trace(go.Box(
                y=res2_clean['total_return_pct'],
                name=strategy2_name,
                boxpoints='outliers'
            ))
            fig5.update_layout(
                title="Box Plot des Rendements",
                yaxis_title="Rendement Total (%)",
                height=chart_height//1.2
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with tab3:
        st.subheader("Analyse des Corrélations")
        
        correlation = np.corrcoef(res1_clean['total_return_pct'], res2_clean['total_return_pct'])[0, 1]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot avec ligne de tendance
            fig6 = px.scatter(
                x=res1_clean['total_return_pct'], 
                y=res2_clean['total_return_pct'],
                title=f"Corrélation des Rendements (r = {correlation:.3f})",
                labels={
                    'x': f'Rendement {strategy1_name} (%)',
                    'y': f'Rendement {strategy2_name} (%)'
                },
                trendline="ols",
                height=chart_height
            )
            fig6.add_shape(
                type="line", line=dict(dash="dash", color="red"),
                x0=min(res1_clean['total_return_pct']), 
                x1=max(res1_clean['total_return_pct']),
                y0=min(res1_clean['total_return_pct']), 
                y1=max(res1_clean['total_return_pct'])
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            st.metric("Corrélation", f"{correlation:.3f}")
            if correlation > 0.7:
                st.success("Corrélation forte positive")
            elif correlation > 0.3:
                st.warning("Corrélation modérée positive")
            elif correlation > -0.3:
                st.info("Corrélation faible")
            else:
                st.error("Corrélation négative")
            
            st.write("**Interprétation:**")
            if abs(correlation) > 0.7:
                st.write("Les stratégies évoluent de manière très similaire")
            elif abs(correlation) > 0.3:
                st.write("Les stratégies ont des mouvements partiellement liés")
            else:
                st.write("Les stratégies évoluent de manière indépendante")
    
    with tab4:
        st.subheader("Comparaisons Détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Valeurs finales
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=res1_clean['final_value'],
                y=res2_clean['final_value'],
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                name='Périodes d\'investissement'
            ))
            min_val = min(res1_clean['final_value'].min(), res2_clean['final_value'].min())
            max_val = max(res1_clean['final_value'].max(), res2_clean['final_value'].max())
            fig7.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Ligne d\'égalité'
            ))
            fig7.update_layout(
                title="Comparaison des Valeurs Finales",
                xaxis_title=f"Valeur Finale - {strategy1_name} ($)",
                yaxis_title=f"Valeur Finale - {strategy2_name} ($)",
                height=chart_height//1.2
            )
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            # Ratios de performance
            fig8 = go.Figure()
            fig8.add_trace(go.Bar(
                x=[strategy1_name, strategy2_name],
                y=[sharpe_1, sharpe_2],
                marker_color=['lightblue', 'lightcoral']
            ))
            fig8.update_layout(
                title="Ratio Risque/Rendement",
                yaxis_title="Ratio (Rendement/Volatilité)",
                height=chart_height//1.2
            )
            st.plotly_chart(fig8, use_container_width=True)
    
    # Section statistiques détaillées
    if show_stats:
        st.markdown("---")
        st.subheader("📊 Statistiques Détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{strategy1_name}**")
            stats_df1 = pd.DataFrame({
                'Métrique': [
                    'Nombre de périodes',
                    'Investissement moyen',
                    'Valeur finale moyenne',
                    'Rendement moyen',
                    'Rendement médian',
                    'Rendement max',
                    'Rendement min',
                    'Volatilité',
                    'Ratio Sharpe'
                ],
                'Valeur': [
                    f"{len(res1_clean)}",
                    f"{res1_clean['total_investment'].mean():,.0f}$",
                    f"{res1_clean['final_value'].mean():,.0f}$",
                    f"{res1_clean['total_return_pct'].mean():.1f}%",
                    f"{res1_clean['total_return_pct'].median():.1f}%",
                    f"{res1_clean['total_return_pct'].max():.1f}%",
                    f"{res1_clean['total_return_pct'].min():.1f}%",
                    f"{res1_clean['total_return_pct'].std():.1f}%",
                    f"{sharpe_1:.2f}"
                ]
            })
            st.dataframe(stats_df1, use_container_width=True)
        
        with col2:
            st.markdown(f"**{strategy2_name}**")
            stats_df2 = pd.DataFrame({
                'Métrique': [
                    'Nombre de périodes',
                    'Investissement moyen',
                    'Valeur finale moyenne',
                    'Rendement moyen',
                    'Rendement médian',
                    'Rendement max',
                    'Rendement min',
                    'Volatilité',
                    'Ratio Sharpe'
                ],
                'Valeur': [
                    f"{len(res2_clean)}",
                    f"{res2_clean['total_investment'].mean():,.0f}$",
                    f"{res2_clean['final_value'].mean():,.0f}$",
                    f"{res2_clean['total_return_pct'].mean():.1f}%",
                    f"{res2_clean['total_return_pct'].median():.1f}%",
                    f"{res2_clean['total_return_pct'].max():.1f}%",
                    f"{res2_clean['total_return_pct'].min():.1f}%",
                    f"{res2_clean['total_return_pct'].std():.1f}%",
                    f"{sharpe_2:.2f}"
                ]
            })
            st.dataframe(stats_df2, use_container_width=True)
    
    # Tests statistiques
    if show_tests:
        st.markdown("---")
        st.subheader("🔬 Tests Statistiques")
        
        # Test de Student
        t_stat, p_value = stats.ttest_ind(res1_clean['total_return_pct'], res2_clean['total_return_pct'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Test t de Student** (comparaison des moyennes)")
            st.metric("Statistique t", f"{t_stat:.3f}")
            st.metric("P-value", f"{p_value:.6f}")
            if p_value < 0.05:
                st.success("✅ Différence significative entre les rendements moyens")
            else:
                st.info("❌ Pas de différence significative entre les rendements moyens")
        
        with col2:
            # Test de Wilcoxon
            try:
                wilcox_stat, wilcox_p = stats.wilcoxon(res1_clean['total_return_pct'] - res2_clean['total_return_pct'])
                st.markdown("**Test de Wilcoxon** (comparaison des médianes)")
                st.metric("Statistique", f"{wilcox_stat:.3f}")
                st.metric("P-value", f"{wilcox_p:.6f}")
            except:
                st.warning("Test de Wilcoxon non applicable (données identiques)")
    
    # Recommandations
    st.markdown("---")
    st.subheader("💡 Recommandations")
    
    avg_outperformance = (res1_clean['total_return_pct'] - res2_clean['total_return_pct']).mean()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if res1_clean['total_return_pct'].mean() > res2_clean['total_return_pct'].mean():
            if res1_clean['total_return_pct'].std() <= res2_clean['total_return_pct'].std():
                st.success(f"✅ **{strategy1_name}** semble supérieure: meilleur rendement avec moins ou égal de risque")
            else:
                st.warning(f"⚠️ **{strategy1_name}** a un meilleur rendement mais plus de risque. Évaluez votre tolérance au risque.")
        else:
            if res2_clean['total_return_pct'].std() <= res1_clean['total_return_pct'].std():
                st.success(f"✅ **{strategy2_name}** semble supérieure: meilleur rendement avec moins ou égal de risque")
            else:
                st.warning(f"⚠️ **{strategy2_name}** a un meilleur rendement mais plus de risque. Évaluez votre tolérance au risque.")
        
        st.write(f"**Performance comparative:**")
        st.write(f"- {strategy1_name} surperforme: {wins_strategy1}/{len(res1_clean)} fois ({win_rate_1:.1f}%)")
        st.write(f"- Surperformance moyenne: {avg_outperformance:.2f}%")
        st.write(f"- Corrélation entre stratégies: {correlation:.3f}")
    
    with col2:
        # Graphique radar des métriques
        categories = ['Rendement', 'Stabilité', 'Ratio Sharpe']
        
        # Normalisation des métriques (0-100)
        max_return = max(res1_clean['total_return_pct'].mean(), res2_clean['total_return_pct'].mean())
        min_vol = min(res1_clean['total_return_pct'].std(), res2_clean['total_return_pct'].std())
        max_sharpe = max(sharpe_1, sharpe_2) if max(sharpe_1, sharpe_2) > 0 else 1
        
        values1 = [
            (res1_clean['total_return_pct'].mean() / max_return * 100) if max_return > 0 else 0,
            (min_vol / res1_clean['total_return_pct'].std() * 100) if res1_clean['total_return_pct'].std() > 0 else 0,
            (sharpe_1 / max_sharpe * 100) if max_sharpe > 0 else 0
        ]
        
        values2 = [
            (res2_clean['total_return_pct'].mean() / max_return * 100) if max_return > 0 else 0,
            (min_vol / res2_clean['total_return_pct'].std() * 100) if res2_clean['total_return_pct'].std() > 0 else 0,
            (sharpe_2 / max_sharpe * 100) if max_sharpe > 0 else 0
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values1,
            theta=categories,
            fill='toself',
            name=strategy1_name,
            opacity=0.6
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values2,
            theta=categories,
            fill='toself',
            name=strategy2_name,
            opacity=0.6
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Profil de Performance",
            height=300
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Retourner les données pour utilisation ultérieure
    return {
        'summary_stats': {
            'strategy1': {
                'mean_return': res1_clean['total_return_pct'].mean(),
                'volatility': res1_clean['total_return_pct'].std(),
                'sharpe_ratio': sharpe_1,
                'win_rate': win_rate_1
            },
            'strategy2': {
                'mean_return': res2_clean['total_return_pct'].mean(),
                'volatility': res2_clean['total_return_pct'].std(),
                'sharpe_ratio': sharpe_2,
                'win_rate': 100 - win_rate_1
            }
        },
        'statistical_tests': {
            't_test': {'statistic': t_stat, 'p_value': p_value}
        },
        'correlation': correlation
    }

