import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Greeting App 😊", page_icon="🌈", layout="centered")

# Header section
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🌟 Welcome to the Greeting App 🌟</h1>", unsafe_allow_html=True)
st.write("### 🧑‍💻 Fill in your details below and get a personalized greeting!")

# Input fields inside a nice container
with st.container():
    name = st.text_input("✍️ Enter your name:")
    age = st.slider("🎂 Select your age:", 1, 100, 25)

# Button section
if st.button("💫 Greet Me!"):
    if name.strip():
        # Choose emoji based on age
        if age < 18:
            emoji = "🧒"
        elif age < 40:
            emoji = "🧑"
        else:
            emoji = "🧓"

        current_hour = datetime.now().hour
        if current_hour < 12:
            greeting_time = "☀️ Good Morning"
        elif current_hour < 18:
            greeting_time = "🌤️ Good Afternoon"
        else:
            greeting_time = "🌙 Good Evening"

        st.success(f"{greeting_time}, {name}! {emoji}\n\n🎉 You are {age} years young and awesome! 😎")
    else:
        st.warning("⚠️ Please enter your name to get a greeting.")

# Footer note
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
