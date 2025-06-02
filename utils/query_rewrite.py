"""
query_rewrite.py

Handles rewriting real estate search queries into clear, constraint-based language using Gemini.
Used to improve the performance of semantic filtering and claim verification.
"""

import streamlit as st
import google.generativeai as palm


@st.cache_resource
def get_gemini_client():

    """
    Initialize and cache Gemini client using the Streamlit secret key.

    Returns:
        palm (google.generativeai): Authenticated client object.
    """
    
    palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    return palm

def rewrite_query_with_constraints(user_query):

    """
    Rewrite a natural language real estate query into structured, factual statements using Gemini.

    Args:
        user_query (str): The original user input.

    Returns:
        str: Reformulated query with clearer constraint and preference structure.
    """

    try:
        client = get_gemini_client()
        prompt = f"""
                You are rewriting a real estate search query to make all preferences and constraints explicit, factual, and broken into individual sentences.

                🧠 Instructions:
                - DO NOT explain or justify anything.
                - DO NOT add extra ideas not in the query.
                - DO NOT repeat the same concept in different words.

                ✅ DO:
                - Use natural language, one statement per sentence.
                - Use strong constraint language like:
                  • "must include", "is required", "must not include", "is prohibited"
                  → for quantifiable constraints (e.g., bedrooms, price, balcony, floor)
                  → not for vague concepts( e.g., center of the city, quite area, in good neighborhood, in a city that's not names explicitly)

                - Use softer language like:
                  • "should preferably", "ideally", "would be nice if"
                  → for vague lifestyle preferences (e.g., quiet, family-friendly)

                - Split combined ideas into separate sentences.
                - Clarify vague terms like:
                  • "near transport" → "should include access to public transportation"
                  • "quiet area" → "should be in a quiet residential neighborhood"

                📥 Input Query: "{user_query}"
                """

        response = client.GenerativeModel("gemini-2.0-flash").generate_content(
            contents=[prompt],
        )
        return response.text.strip()
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return user_query  # fallback
