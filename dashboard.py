import streamlit as st
import sqlite3
import pandas as pd

conn = sqlite3.connect("database/customers.db")

df = pd.read_sql_query(
    "SELECT * FROM customers",
    conn
)

st.title("Customer Analytics")

st.dataframe(df)