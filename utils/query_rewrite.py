# utils/query_rewrite.py

import streamlit as st
import google.generativeai as palm

# Load Gemini client using Streamlit secrets
@st.cache_resource
def get_gemini_client():
    palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return palm

def rewrite_query_with_constraints(user_query):
    try:
        client = get_gemini_client()
        prompt = f"""
        Rewrite the following real estate search query to make all preferences and constraints explicit.

        Rules:
        - Do not add any explanation or extra text
        - Just return the rewritten query
        - Use natural language
        - Use strong constraint phrases like \"must include\", \"must not have\", \"is required\", or \"cannot include\"
        - Write each constraint as a separate sentence
        - Do not repeat the same idea
        - Avoid technical language like \"floor number below the highest\"

        Query: \"{user_query}\"
        """
        response = client.GenerativeModel("gemini-2.0-flash").generate_content(
            contents=[prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return user_query  # fallback
