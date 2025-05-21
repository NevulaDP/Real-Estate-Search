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
        Rewrite the following real estate search query to make all preferences and constraints explicit and phrased as strict requirements.
        
        Guidelines:
        - Use short, clear natural language statements.
        - Each constraint must be a standalone sentence.
        - Use **strong formulations** like:
            - "The property is required to be located in..."
            - "The price must be under..."
            - "The apartment must include..."
            - "The apartment cannot have..."
        - Avoid weak phrases like “I want” or “should be”.
        - Do not rephrase or simplify user intent — keep it strict.
        - Do not add any explanations or extra text.
        
        Query: \"{user_query}\"
        """
        response = client.GenerativeModel("gemini-2.0-flash").generate_content(
            contents=[prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return user_query  # fallback
