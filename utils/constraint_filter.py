# utils/constraint_filter.py

import re
from utils.nli_filter import split_query

def extract_constraints_from_query(query):
    query = query.lower()
    constraints = {}

    # Convert word numbers to digits
    word_to_number = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    for word, digit in word_to_number.items():
        query = query.replace(word, digit)

    # Ex. Convert "2.5 million" to 2500000
    query = re.sub(
        r'(\d+(?:\.\d+)?)\s*million',
        lambda m: str(int(float(m.group(1)) * 1_000_000)),
        query
    )

    # --- PRICE ---
    price_pattern = r'(?:price|dollars|\$)?\s*(?:under|less than|below|priced under|costs less than|not exceed|cannot exceed)[^\d]{0,10}(\d[\d,\.]*)'
    match_max_price = re.search(price_pattern, query)
    if match_max_price:
        try:
            constraints["max_price"] = int(float(match_max_price.group(1).replace(',', '')))
        except:
            pass

    price_min_pattern = r'(?:price|dollars|\$)?\s*(?:more than|at least|over|minimum price)[^\d]{0,10}(\d[\d,\.]*)'
    match_min_price = re.search(price_min_pattern, query)
    if match_min_price:
        try:
            constraints["min_price"] = int(float(match_min_price.group(1).replace(',', '')))
        except:
            pass

    # --- SIZE --- (only if unit is present)
    size_unit = r'(?:\s*(?:ft|feet|sq|square(?: feet)?|sqm|meters))'

    match_max_size = re.search(
        rf'(?:less than|under|smaller than|max(?:imum)? size|below)\s*(\d{{2,5}}){size_unit}',
        query
    )
    if match_max_size:
        constraints["max_size"] = int(match_max_size.group(1))

    match_min_size = re.search(
        rf'(?:more than|greater than|larger than|at least|min(?:imum)? size|above)\s*(\d{{2,5}}){size_unit}',
        query
    )
    if match_min_size:
        constraints["min_size"] = int(match_min_size.group(1))

    # --- BEDROOMS ---
    match_min_beds = re.search(r'(?:at least|minimum of|more than)[^\d]{0,15}(\d+)[^\w]{0,5}(?:bedroom|bedrooms)', query)
    if match_min_beds:
        constraints["min_bedrooms"] = int(match_min_beds.group(1))

    match_max_beds = re.search(r'(?:maximum of|less than|under)\s*(\d+)\s*(?:bedroom|bedrooms)', query)
    if match_max_beds:
        constraints["max_bedrooms"] = int(match_max_beds.group(1))

    # --- BATHROOMS ---
    match_min_baths = re.search(r'(?:at least|a minimum of|minimum of|more than)\s*(\d+)\s*(?:bathroom|bathrooms)', query)
    if match_min_baths:
        constraints["min_bathrooms"] = int(match_min_baths.group(1))

    match_max_baths = re.search(r'(?:maximum of|a maximum of|less than|under)\s*(\d+)\s*(?:bathroom|bathrooms)', query)
    if match_max_baths:
        constraints["max_bathrooms"] = int(match_max_baths.group(1))

    return constraints

def apply_constraint_filters(data, constraints):
    filtered = data
    if "max_price" in constraints:
        filtered = [d for d in filtered if d.get("price", 0) < constraints["max_price"]]
    if "min_price" in constraints:
        filtered = [d for d in filtered if d.get("price", 0) > constraints["min_price"]]
    if "max_size" in constraints:
        filtered = [d for d in filtered if d.get("size", 0) < constraints["max_size"]]
    if "min_size" in constraints:
        filtered = [d for d in filtered if d.get("size", 0) > constraints["min_size"]]
    if "min_bedrooms" in constraints:
        filtered = [d for d in filtered if d.get("num_bedrooms", 0) >= constraints["min_bedrooms"]]
    if "max_bedrooms" in constraints:
        filtered = [d for d in filtered if d.get("num_bedrooms", 0) <= constraints["max_bedrooms"]]
    if "min_bathrooms" in constraints:
        filtered = [d for d in filtered if d.get("num_bathrooms", 0) >= constraints["min_bathrooms"]]
    if "max_bathrooms" in constraints:
        filtered = [d for d in filtered if d.get("num_bathrooms", 0) <= constraints["max_bathrooms"]]
    return filtered

def filter_semantic_subqueries(rewritten, constraints):
    subqueries = split_query(rewritten)

    # Ensure subqueries is a list
    if isinstance(subqueries, str):
        subqueries = [subqueries]

    keywords = ["price", "size", "square", "million", "$", "ft", "feet", "bedroom", "bathroom"]

    filtered = []
    for q in subqueries:
        if not any(kw in q.lower() for kw in keywords):
            print("✅ Keeping subquery:", q)  # Debug line
            filtered.append(q.strip())
    return filtered
