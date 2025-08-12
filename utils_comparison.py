# utils_comparison.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from scipy import stats

# Importer la fonction de calcul depuis le module de l'analyse simple
# This assumes utils_single.py is in the same directory.
from utils_single import calculate_dca_metrics

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


def display_strategy_comparison(res1, res2, name1, name2):
    """
    Affiche une comparaison riche et détaillée de deux stratégies d'investissement,
    avec des onglets, des graphiques avancés et des tests statistiques.
    """
    st.header(f"⚖️ Comparaison Détaillée : {name1} vs. {name2}")

    if res1 is None or res2 is None or res1.empty or res2.empty:
        st.error("Les données de comparaison sont manquantes ou incomplètes.")
        return

    # --- Préparation et Nettoyage des Données ---
    def clean_data(df):
        df_clean = df.copy()
        for col in ['total_return_pct', 'annualized_return', 'final_value', 'total_investment']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean = df_clean.dropna().sort_index()
        return df_clean
    
    res1_clean = clean_data(res1)
    res2_clean = clean_data(res2)
    
    common_index = res1_clean.index.intersection(res2_clean.index)
    res1_clean = res1_clean.loc[common_index]
    res2_clean = res2_clean.loc[common_index]

    if res1_clean.empty or res2_clean.empty:
        st.error("Aucune période commune valide trouvée après le nettoyage des données.")
        return

    # --- Métriques Principales ---
    col1, col2, col3, col4 = st.columns(4)
    
    mean_return_1 = res1_clean['annualized_return'].mean()
    mean_return_2 = res2_clean['annualized_return'].mean()
    std_dev_1 = res1_clean['annualized_return'].std()
    std_dev_2 = res2_clean['annualized_return'].std()
    sharpe_1 = mean_return_1 / std_dev_1 if std_dev_1 > 0 else 0
    sharpe_2 = mean_return_2 / std_dev_2 if std_dev_2 > 0 else 0
    win_rate_1 = (res1_clean['final_value'] > res2_clean['final_value']).sum() / len(res1_clean) * 100
    
    col1.metric("Rdt Ann. Moyen", f"{mean_return_1:.1f}%", f"{mean_return_1 - mean_return_2:.1f}% vs {name2}")
    col2.metric("Volatilité Ann.", f"{std_dev_1:.1f}%", f"{std_dev_1 - std_dev_2:.1f}% vs {name2}")
    col3.metric(f"Taux de Victoire ({name1})", f"{win_rate_1:.1f}%", help=f"Pourcentage de périodes où {name1} a surperformé {name2}.")
    col4.metric("Ratio de Sharpe", f"{sharpe_1:.2f}", f"{sharpe_1 - sharpe_2:.2f} vs {name2}")

    st.markdown("---")

    # --- Onglets pour Organiser les Analyses ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Performance", "📊 Distributions", "🔗 Corrélation", "📝 Statistiques", "💡 Synthèse"])
    
    with tab1:
        st.subheader("Évolution de la Performance sur les Périodes Glissantes")
        
        perf_diff = res1_clean['final_value'] - res2_clean['final_value']
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(
            x=perf_diff.index, y=perf_diff, mode='lines', name=f'Différence ({name1} - {name2})',
            line=dict(width=2, color='grey'), fill='tozeroy',
            fillcolor='rgba(0, 200, 0, 0.1)',
        ))
        fig_diff.add_hline(y=0, line_dash="dash", line_color="red")
        fig_diff.update_layout(
            title=f"Surperformance de {name1} vs {name2} (Valeur Finale)",
            xaxis_title="Date de Début de la Période d'Investissement",
            yaxis_title="Différence de Valeur ($)",
            height=400, hovermode='x unified'
        )
        st.plotly_chart(fig_diff, use_container_width=True)
        
    with tab2:
        st.subheader("Analyse des Distributions de Rendement")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=res1_clean['annualized_return'], name=name1, opacity=0.7, nbinsx=20))
            fig_hist.add_trace(go.Histogram(x=res2_clean['annualized_return'], name=name2, opacity=0.7, nbinsx=20))
            fig_hist.update_layout(title="Distribution des Rendements Annualisés", barmode='overlay', height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=res1_clean['annualized_return'], name=name1, boxpoints='outliers'))
            fig_box.add_trace(go.Box(y=res2_clean['annualized_return'], name=name2, boxpoints='outliers'))
            fig_box.update_layout(title="Box Plot des Rendements Annualisés", yaxis_title="Rendement (%)", height=400)
            st.plotly_chart(fig_box, use_container_width=True)
            
    with tab3:
        st.subheader("Analyse de Corrélation")
        correlation = res1_clean['annualized_return'].corr(res2_clean['annualized_return'])
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # === CODE CORRIGÉ ICI ===
            fig_corr = px.scatter(
                x=res1_clean['annualized_return'], y=res2_clean['annualized_return'],
                trendline="ols", # trendline_color argument removed from here
                labels={'x': f'Rendement {name1} (%)', 'y': f'Rendement {name2} (%)'}
            )
            # This is the correct way to style the trendline
            fig_corr.update_traces(selector=dict(type='scatter', mode='lines'), line=dict(color='red'))
            
            fig_corr.update_layout(title=f"Corrélation des Rendements (r = {correlation:.3f})", height=450)
            st.plotly_chart(fig_corr, use_container_width=True)
            # === FIN DE LA CORRECTION ===
        with col2:
            st.metric("Coefficient de Corrélation", f"{correlation:.3f}")
            if correlation > 0.7: st.info("Corrélation forte : les stratégies évoluent de manière très similaire.")
            elif correlation > 0.3: st.info("Corrélation modérée : les stratégies ont des mouvements partiellement liés.")
            else: st.info("Corrélation faible : les stratégies évoluent de manière assez indépendante.")

    with tab4:
        st.subheader("Statistiques Détaillées et Tests")
        col1, col2 = st.columns(2)
        
        def get_stats_df(df, name):
            sharpe = df['annualized_return'].mean() / df['annualized_return'].std() if df['annualized_return'].std() > 0 else 0
            return pd.DataFrame({
                'Métrique': ['Nb Périodes', 'Rdt Ann. Moyen', 'Rdt Ann. Médian', 'Volatilité Ann.', 'Rdt Min', 'Rdt Max', 'Sharpe Ratio'],
                'Valeur': [
                    f"{len(df)}", f"{df['annualized_return'].mean():.2f}%", f"{df['annualized_return'].median():.2f}%",
                    f"{df['annualized_return'].std():.2f}%", f"{df['annualized_return'].min():.2f}%", f"{df['annualized_return'].max():.2f}%",
                    f"{sharpe:.2f}"
                ]
            }).set_index('Métrique')
            
        with col1:
            st.markdown(f"**{name1}**")
            st.dataframe(get_stats_df(res1_clean, name1), use_container_width=True)

        with col2:
            st.markdown(f"**{name2}**")
            st.dataframe(get_stats_df(res2_clean, name2), use_container_width=True)

        st.markdown("**Test Statistique (Test T)**")
        t_stat, p_value = stats.ttest_ind(res1_clean['annualized_return'], res2_clean['annualized_return'], nan_policy='omit')
        
        scol1, scol2 = st.columns(2)
        scol1.metric("P-value", f"{p_value:.4f}", help="Une p-value < 0.05 suggère une différence statistiquement significative entre les moyennes.")
        
        with scol2:
            if p_value < 0.05:
                st.success("La différence de performance est statistiquement significative.")
            else:
                st.warning("La différence n'est PAS statistiquement significative.")

    with tab5:
        st.subheader("Synthèse et Recommandation")
        col1, col2 = st.columns([1, 1])

        with col1:
            categories = ['Rendement', 'Stabilité (Inv. Vol)', 'Ratio Sharpe']
            
            max_ret = max(mean_return_1, mean_return_2) if max(mean_return_1, mean_return_2) != 0 else 1
            min_vol = min(std_dev_1, std_dev_2) if min(std_dev_1, std_dev_2) != 0 else 1
            max_sha = max(sharpe_1, sharpe_2) if max(sharpe_1, sharpe_2) != 0 else 1

            val1 = [mean_return_1/max_ret, min_vol/std_dev_1 if std_dev_1 > 0 else 0, sharpe_1/max_sha]
            val2 = [mean_return_2/max_ret, min_vol/std_dev_2 if std_dev_2 > 0 else 0, sharpe_2/max_sha]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=val1, theta=categories, fill='toself', name=name1))
            fig_radar.add_trace(go.Scatterpolar(r=val2, theta=categories, fill='toself', name=name2))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Profil de Performance Relatif", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown("**Conclusion**")
            if mean_return_1 > mean_return_2:
                if std_dev_1 <= std_dev_2:
                    st.success(f"✅ **{name1}** semble supérieur : il offre un meilleur rendement moyen pour un risque plus faible ou égal.")
                else:
                    st.warning(f"⚠️ **{name1}** offre un meilleur rendement mais avec un risque plus élevé. Un choix pour les investisseurs tolérants au risque.")
            else:
                 if std_dev_2 <= std_dev_1:
                    st.success(f"✅ **{name2}** semble supérieur : il offre un meilleur rendement moyen pour un risque plus faible ou égal.")
                 else:
                    st.warning(f"⚠️ **{name2}** offre un meilleur rendement mais avec un risque plus élevé.")
            
            st.markdown(f"Sur les **{len(common_index)}** périodes analysées, **{name1}** a surperformé **{win_rate_1:.1f}%** du temps.")