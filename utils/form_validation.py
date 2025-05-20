def validate_and_process_form(title, short_description, location, price, size,
                              num_bedrooms, num_bathrooms, floor, uploaded_images,
                              balcony, parking, extract_features_func, model):
    errors = []

    if not title.strip():
        errors.append("Title is required.")
    if not short_description.strip():
        errors.append("Short description is required.")
    if not location.strip():
        errors.append("Location is required.")
    if price <= 0:
        errors.append("Price must be greater than 0.")
    if size <= 0:
        errors.append("Size must be greater than 0.")
    if num_bedrooms < 0:
        errors.append("Number of bedrooms cannot be negative.")
    if num_bathrooms < 0:
        errors.append("Number of bathrooms cannot be negative.")
    if floor < 0:
        errors.append("Floor number cannot be negative.")
    if not uploaded_images:
        errors.append("At least one image is required.")

    if errors:
        return None, errors

    all_features = []
    for image_file in uploaded_images:
        items = extract_features_func(image_file, model)
        all_features.extend(items)

    inputs = {
        "title": title,
        "short_description": short_description,
        "location": location,
        "price": price,
        "size": size,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "balcony": balcony,
        "parking": parking,
        "floor": floor,
        "uploaded_images": uploaded_images
    }

    return (all_features, inputs), None
