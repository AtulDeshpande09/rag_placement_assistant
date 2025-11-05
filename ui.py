import streamlit as st
from your_file_name import generate_interview_response   # replace with your actual filename

st.set_page_config(page_title="Placement Prep Assistant", layout="centered")

st.title("🎯 AI Placement Preparation Assistant")

query = st.text_input("Enter your Query", placeholder="e.g., Interview questions for Infosys Python Developer")

if st.button("Generate"):
    if query.strip() == "":
        st.error("Please enter a query.")
    else:
        with st.spinner("Thinking..."):
            answer = generate_interview_response(query)
        st.write("### ✅ Response")
        st.write(answer)
