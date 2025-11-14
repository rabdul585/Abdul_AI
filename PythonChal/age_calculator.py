import streamlit as st
from datetime import date

# App title
st.title("🎂 Simple Age Calculator")

# Input: Date of birth
dob = st.date_input("Enter your Date of Birth", value=date(2000, 1, 1), max_value=date.today())

# Calculate age
today = date.today()
age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# Display result
st.write(f"👶 You are **{age} years old** as of today ({today}).")
