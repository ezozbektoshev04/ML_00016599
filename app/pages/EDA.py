import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/serie_a_merged.csv")
df.drop(columns=["Referee"], inplace=True)
st.title("Exploratory Data Analysis")

#Dataset Preview 
st.subheader("Dataset Preview")
st.dataframe(df.head(20))  

# Shape information 
st.markdown(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("Home Goals Distribution")
fig, ax = plt.subplots()
df["FTHG"].hist(bins=20, ax=ax)
st.pyplot(fig)

st.subheader("Away Goals Distribution")
fig, ax = plt.subplots()
df["FTAG"].hist(bins=20, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Matrix")

corr = df[["FTHG", "FTAG", "HS", "AS", "HST", "AST"]].corr()

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
