# from flow import run_pipeline

# while True:
#     user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
#     if user_query.lower() == "exit":
#         print("Exiting the program.")
#         break

#     print("\nProcessing your query...")
#     result = run_pipeline(user_query)
    
#     print("\nResponse:\n", result)

import streamlit as st
import tempfile
from flow import run_pipeline
# from config import create_llm
# import config

st.title("LLM Query App with PDF Upload")

# groq_api_key = st.text_input("Enter your groq_api_key", type="password")

model = st.selectbox("Choose model", options=["gemma2-9b-it", "another-model"])

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

user_query = st.text_area("Enter your query")

if st.button("Submit"):
    if user_query and uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_pdf_path = tmp_file.name

        # config.llm = create_llm(groq_api_key, model)

        with st.spinner("Processing your query..."):
            result = run_pipeline(user_query,pdf_path=tmp_pdf_path)  
        st.subheader("Response")
        st.write(result)
    else:
        st.error("Please provide your groq_api_key, upload a PDF, and enter a query.")