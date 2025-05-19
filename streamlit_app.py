from utils.database import create_property_entry

# Generate embedding
embedding = model.encode(final_text)

# Create new entry
new_entry = create_property_entry(
    title, short_description, location, price, size,
    num_bedrooms, num_bathrooms, balcony, parking, floor,
    all_features, embedding, [img.name for img in uploaded_images]
)

# In-memory DB (for now)
if "entries" not in st.session_state:
    st.session_state.entries = []

st.session_state.entries.append(new_entry)

st.success("✅ Entry saved (in session)")
st.write("🧾 Current Entries:", len(st.session_state.entries))
st.json(new_entry)

