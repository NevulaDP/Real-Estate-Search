# utils/constraint_filter.py

import re
from utils.inferring_filter import split_query

def extract_constraints_from_query(query):

    """
    Extracts structured constraints (price, size, bedrooms, bathrooms, location) from a natural language query.
    
    Args:
        query (str): User's raw query text.
    
    Returns:
        dict: A dictionary of extracted constraints.
    """

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
    price_pattern = r'(?:price|dollars|\$)?\s*(?:under|less than|below|priced under|costs less than|not exceed|cannot exceed|cost no more than|not cost more than)[^\d]{0,10}(\d[\d,\.]*)'
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
    size_unit = r'(?:\s*(?:ft|feet|sq|square feet|square(?: feet)?|sqm|meters))'

    match_max_size = re.search(
        rf'(?:less than|under|smaller than|max(?:imum)? size|below)\s*(\d{{2,5}}){size_unit}',
        query
    )
    if match_max_size:
        constraints["max_size"] = int(match_max_size.group(1))

    match_min_size = re.search(
        rf'(?:more than|greater than|larger than|at least|over|min(?:imum)? size|above)\s*(\d{{2,5}}){size_unit}',
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
        
    # --- Exact number of bedrooms ---
    match_exact_beds = re.search(r'\b(?:must have|has|have|offers|include|includes)\s+(\d+)\s+(?:bedroom|bedrooms)\b', query)
    if match_exact_beds:
        constraints["min_bedrooms"] = int(match_exact_beds.group(1))
        constraints["max_bedrooms"] = int(match_exact_beds.group(1))

    # --- Exact number of bathrooms ---
    match_exact_baths = re.search(r'\b(?:must have|has|have|offers|include|includes)\s+(\d+)\s+(?:bathroom|bathrooms)\b', query)
    if match_exact_baths:
        constraints["min_bathrooms"] = int(match_exact_baths.group(1))
        constraints["max_bathrooms"] = int(match_exact_baths.group(1))
    
    # --- Location ---
    location_match = re.search(
    r'\b(?:located\s+in|(?:property|apartment|house|penthouse|unit)\s+(?:must\s+be\s+)?in)\s+([a-zA-Z\s]+)',
    query
    )
    if location_match:
        location = location_match.group(1).strip().lower()
        constraints["location"] = location

    return constraints

def apply_constraint_filters(data, constraints):

    """
    Applies structured constraints to a list of property entries.
    
    Args:
        data (list): List of property dictionaries.
        constraints (dict): Constraint values to filter by.
    
    Returns:
        list: Filtered list of properties that match the constraints.
    """

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
    if "location" in constraints:
        loc = constraints["location"]
        filtered = [d for d in filtered if d.get("location", "").lower() == loc]
    return filtered

    
def filter_semantic_subqueries(rewritten: str, constraints: dict) -> list:

    """
    Filters rewritten subqueries to remove those that only contain structured constraints.
    
    Args:
        rewritten (str): Full rewritten query.
        constraints (dict): Detected structured constraints.
    
    Returns:
        list: Subqueries that represent unstructured or semantic information.
    """

    subqueries = split_query(rewritten)
    filtered = [q.strip() for q in subqueries if not is_structured_constraint(q)]

    return filtered



def is_structured_constraint(phrase: str) -> bool:

    """
    Determines whether a phrase is a structured constraint (e.g., size, price).
    
    Args:
        phrase (str): Subquery or phrase to evaluate.
    
    Returns:
        bool: True if it's a structured constraint, False otherwise.
    """

    phrase = phrase.lower()

    price_pattern = r"(price|cost|under|less than|below|million|₪|\d+\s*(m|million|₪))"
    size_pattern = r"(size|square meters|sqm|larger than|smaller than|more than|less than|\d+\s*(sqm|m²))"
    bedroom_pattern = r"(bedroom|room|sleeping area)"
    bathroom_pattern = r"(bathroom|toilet|restroom)"
    location_pattern = r"(in|located in|neighborhood|area)\s+\b(tel aviv|jerusalem|netanya|beersheba|haifa)\b"

    patterns = [
        price_pattern,
        size_pattern,
        bedroom_pattern,
        bathroom_pattern,
        location_pattern,
    ]

    return any(re.search(p, phrase) for p in patterns)

