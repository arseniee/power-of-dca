import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import datetime 
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib.ticker import FuncFormatter
from utils import *
from datetime import date


# Most Famous ETFS
etf_list =  {
    "SPDR S&P 500 ETF Trust": "SPY",
    "3x Daily L Nasdaq-100": 'QQQ3.MI',
    "iShares Core S&P 500 ETF": "IVV",
    "Vanguard S&P 500 ETF": "VOO",
    "Invesco QQQ Trust": "QQQ",
    "Vanguard Total Stock Market ETF": "VTI",
    "iShares Russell 2000 ETF": "IWM",
    "SPDR Gold Trust": "GLD",
    "iShares MSCI Emerging Markets ETF": "EEM",
    "iShares MSCI EAFE ETF": "EFA",
    "iShares 20+ Year Treasury Bond ETF": "TLT",
    "iShares iBoxx $ High Yield Corporate Bond ETF": "HYG",
    "Financial Select Sector SPDR Fund": "XLF",
    "Technology Select Sector SPDR Fund": "XLK",
    "Health Care Select Sector SPDR Fund": "XLV",
    "Consumer Discretionary Select Sector SPDR Fund": "XLY",
    "Energy Select Sector SPDR Fund": "XLE",
    "Utilities Select Sector SPDR Fund": "XLU",
    "iShares Silver Trust": "SLV",
    "ARK Innovation ETF": "ARKK",
    "Schwab U.S. Dividend Equity ETF": "SCHD",
    "Vanguard Dividend Appreciation ETF": "VIG",
    "Vanguard Growth ETF": "VUG",
    "Vanguard Value ETF": "VTV",
    "Schwab U.S. Broad Market ETF": "SCHB",
    "iShares Core S&P Total U.S. Stock Market ETF": "ITOT",
    "SPDR Portfolio S&P 500 ETF": "SPLG",
    "ARK Next Generation Internet ETF": "ARKW",
    "Vanguard Information Technology ETF": "VGT",
    "VanEck Gold Miners ETF": "GDX",
    "iShares China Large-Cap ETF": "FXI",
    "iShares Bitcoin Trust": "IBIT",
    "ProShares UltraPro QQQ": "TQQQ",
    "ProShares UltraPro Short QQQ": "SQQQ",
    "SPDR Portfolio Long Term Corporate Bond ETF": "SPLB",
    "iShares Broad USD High Yield Corporate Bond ETF": "USHY",
    "iShares Core Total International Stock ETF": "IXUS",
    "Vanguard FTSE All-World ex-US ETF": "VEU",
    "Vanguard Total World Stock ETF": "VT",
}


# Inputs
st.title("Power of DCA")
compareStocks = st.selectbox("You want to comapre 2 stocks", [False, True], key='0')


if compareStocks:
    # inputs
    investmentPeriods = st.selectbox("Years you want to invest", [1, 2, 3, 4, 5], key='1') * 12
    stock1Name = st.selectbox("Which ETF 1", list(etf_list.keys()), key='2')
    stock2Name = st.selectbox("Which ETF 1", list(etf_list.keys()), key='3')
    monthlyInvestment = st.number_input("Monthly invesement ( $ / month)", value=100, key='4')

    stock1 = get_stock_data(etf_list[stock1Name])
    stock2 = get_stock_data(etf_list[stock2Name])
    res1, res2 = compare_strats(investmentPeriods, stock1, stock2, monthlyInvestment)
    compare_investment_strategies_streamlit(res1, res2)
    



else:
    # inputs
    startDate = st.date_input( "Choose invesment start date", value=date(2010, 1, 1), key='10')
    endDate = st.date_input( "Choose invesment end date",value=date.today(), key='11')
    stock1Name = st.selectbox("Which ETF ?", list(etf_list.keys()), key='12')
    monthlyInvestment = st.number_input("Monthly invesement ( $ / month)", value=100, key='13')


    stock1 = get_stock_data(etf_list[stock1Name])
    test_strat_graphs(startDate, endDate, stock1, monthlyInvestment)




