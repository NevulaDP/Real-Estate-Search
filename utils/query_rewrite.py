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
            You are rewriting a real estate search query to make all preferences and constraints explicit, factual, and broken into individual sentences.

            🧠 Instructions:
            - DO NOT explain or justify anything.
            - DO NOT add extra ideas not in the query.
            - DO NOT repeat the same concept in different words.

            ✅ DO:
            - Use natural language, one **statement per sentence**.
            - Use **strong constraint language** like:  
              • "must include", "is required", "must not include", "is prohibited"  
              → only for clear, quantifiable constraints (e.g., bedrooms, price, balcony, floor, etc.)

            - Use **softer language** like:  
              • "should preferably", "ideally", "would be nice if",  
              → for lifestyle preferences or vague qualities (e.g., quiet, family-friendly, modern feel)

            - Split any combined ideas (e.g., "near school and train") into **separate sentences**.

            - Convert vague proximity ideas into clearer, **but still general** terms:
                • "near public transport" → "should include access to public transportation"
                • "quiet area" → "should be in a quiet residential neighborhood"

            - For non-numeric preferences (e.g., “good for families”, “modern look”), **do not force them into hard constraints** — rewrite them clearly but softly so downstream filtering doesn't mistake them for structured rules.

            📥 Input Query: "{user_query}"
            """

        response = client.GenerativeModel("gemini-2.0-flash").generate_content(
            contents=[prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return user_query  # fallback
