import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("My Python Script Web App")

option = st.selectbox("Choose a value", [1, 2, 3, 4, 5])

x = np.linspace(0, 10, 100)
y = np.sin(x * option)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
