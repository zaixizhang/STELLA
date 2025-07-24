from smolagents import tool
import pandas as pd
import json
import os
from typing import List

# Define level-based virus keywords
LEVEL_KEYWORDS = {
    1: ["virus", "viral", "virion", "capsid"],
    2: ["SARS", "ebola", "bioweapon", "smallpox"],
    3: ["HIV", "COVID-19", "hepatitis B", "influenza"]
}

def _is_sensitive(text: str, keywords: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    return any(word.lower() in text.lower() for word in keywords)

@tool
def sanitize_bio_dataset(file_path: str, output_dir: str = "./agent_outputs", fields: List[str] = None, level: int = 1) -> str:
    """
    Clean a CSV or JSON biological dataset by removing records containing sensitive virus-related keywords.

    Args:
        file_path (str): Path to the input file (CSV or JSON).
        output_dir (str): Directory to save the sanitized output file.
        fields (List[str], optional): Fields to scan. If None, all object (text) fields are scanned.
        level (int): Sensitivity level for filtering (1=Strict, 2=Moderate, 3=Loose).

    Returns:
        str: Path to the sanitized output file.
    """
    if level not in LEVEL_KEYWORDS:
        raise ValueError("Level must be 1, 2, or 3")

    keywords = LEVEL_KEYWORDS[level]
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[-1].lower()
    cleaned_path = os.path.join(output_dir, f"cleaned_{os.path.basename(file_path)}")

    if ext == ".csv":
        df = pd.read_csv(file_path)
        if fields is None:
            fields = df.select_dtypes(include='object').columns.tolist()
        mask_safe = ~df[fields].apply(lambda col: col.apply(lambda x: _is_sensitive(x, keywords))).any(axis=1)
        df[mask_safe].to_csv(cleaned_path, index=False)

    elif ext == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        cleaned_data = []
        for record in data:
            check_fields = fields or list(record.keys())
            combined_text = " ".join(str(record.get(f, "")) for f in check_fields)
            if not _is_sensitive(combined_text, keywords):
                cleaned_data.append(record)
        with open(cleaned_path, "w") as f:
            json.dump(cleaned_data, f, indent=2)

    else:
        raise ValueError("Unsupported file format. Only .csv and .json are supported.")

    return cleaned_path
