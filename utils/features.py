from PIL import Image
import io
import base64
import json

def extract_features(image_file, client):
    """Extract visual features from an image using Gemini and return confirmed ones."""
    try:
        image = Image.open(image_file).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        Analyze the following image and identify key household items beneficial for buyers.
        Return a JSON array of objects with 'item' and 'description' keys.
        Image data: {img_str}
        """

        response = client.generate_text(prompt=prompt)
        return json.loads(response.text)

    except Exception as e:
        return []

def generate_combined_text(
    title,
    short_description,
    location,
    price,
    size,
    num_bedrooms,
    num_bathrooms,
    balcony,
    parking,
    floor,
    detected_features=None
):
    balcony_text = "This apartment includes a balcony." if balcony else "This apartment does not include a balcony."
    parking_text = "It includes a parking spot." if parking else "This apartment does not include a parking spot."

    if floor == 0:
        floor_text = "The apartment is on the ground floor."
    elif floor == 1:
        floor_text = "The apartment is on the first floor."
    else:
        floor_text = f"The apartment is on the {floor}th floor."

    base_description = (
        f"{title}. {short_description}. "
        f"This property is located in {location}. "
        f"It is priced at {price} dollars and spans {size} square feet. "
        f"It includes {num_bedrooms} bedrooms and {num_bathrooms} bathrooms. "
        f"{balcony_text} {parking_text} {floor_text}"
    )

    feature_descriptions = ""
    if detected_features:
        for feature in detected_features:
            item = feature.get("item", "").strip()
            desc = feature.get("description", "").strip()
            if item and desc:
                feature_descriptions += f" {item}: {desc}"

    return f"{base_description} {feature_descriptions.strip()}"
