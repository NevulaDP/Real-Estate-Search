"""
database.py

Utility for packaging property data into a consistent dictionary format.
Used during property uploads before pushing to a database or Hub.

Assumes all fields are pre-validated and embedding is a NumPy array.
"""

import numpy as np

def create_property_entry(
    property_id,
    title, short_description, location, price, size,
    num_bedrooms, num_bathrooms, balcony, parking, floor,
    detected_features, embedding, image_urls
):

    """
    Create a standardized property entry dictionary.

    Args:
        property_id (str): Unique identifier for the property.
        title (str): Title of the property listing.
        short_description (str): Brief text description.
        location (str): City or region.
        price (int): Price in USD.
        size (int): Size in square feet.
        num_bedrooms (int): Number of bedrooms.
        num_bathrooms (int): Number of bathrooms.
        balcony (bool): Whether the unit has a balcony.
        parking (bool): Whether the unit has parking.
        floor (int): Floor number.
        detected_features (List[Dict]): Structured features from image analysis.
        embedding (np.ndarray): Semantic embedding of the listing.
        image_urls (List[str]): Paths or URLs to uploaded images.

    Returns:
        dict: Full property record to be stored or uploaded.
    """
    return {
        'id': property_id,
        'image_paths': image_urls,
        'title': title,
        'short_description': short_description,
        'location': location,
        'price': price,
        'size': size,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'balcony': balcony,
        'parking': parking,
        'floor': floor,
        'detected_features': detected_features,
        'embedding': embedding.tolist(),
        'combined_text': None, # filled in upload_section utilizing generate_xxx funcs 
        'short_text': None, # filled in upload_section utilizing generate_xxx funcs 
        'semantic_text': None # filled in upload_section utilizing generate_xxx funcs 
    }

