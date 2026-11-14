
# _core_guard/keyword_filter.py
import re
from rapidfuzz import fuzz

# 示例关键词（真实部署时建议用更全的表）
KEYWORDS = [
    "HIV", "retrovirus", "SARS", "MERS", "toxin", "bioweapon", "gain of function"
]

def keyword_check(text: str, threshold: int = 85) -> bool:
    text_lower = text.lower()
    for keyword in KEYWORDS:
        if keyword.lower() in text_lower:
            return True
        if fuzz.partial_ratio(text_lower, keyword.lower()) >= threshold:
            return True
    return False
