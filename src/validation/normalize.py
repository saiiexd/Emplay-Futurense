import re
from typing import Any, List
from src.schemas.candidates import NormalizedValue

def normalize_identifier(value: str) -> NormalizedValue:
    cleaned = re.sub(r'[\W_]+', '', value).upper()
    return NormalizedValue(value=cleaned, data_type="identifier")

def normalize_date(value: str) -> NormalizedValue:
    if re.search(r'(?i)within\s+\d+\s+days', value) or re.search(r'(?i)after\s+award', value):
        return NormalizedValue(value=value, data_type="relative_date")
        
    norm_val = value
    tz_map = {
        "CST": "America/Chicago",
        "CDT": "America/Chicago",
        "EST": "America/New_York",
        "EDT": "America/New_York"
    }
    for tz_str, tz_iana in tz_map.items():
        if tz_str in norm_val:
            norm_val = norm_val.replace(tz_str, tz_iana)
            
    return NormalizedValue(value=norm_val, data_type="datetime")

def normalize_contact(value: dict) -> NormalizedValue:
    norm = dict(value)
    if norm.get("email"):
        norm["email"] = re.sub(r'\s+', '', norm["email"])
    if norm.get("phone"):
        norm["phone"] = re.sub(r'\s+', '', norm["phone"])
    return NormalizedValue(value=norm, data_type="contact")

def normalize_list(values: List[str]) -> NormalizedValue:
    seen = set()
    deduped = []
    for v in values:
        cleaned = re.sub(r'\s+', ' ', v).strip()
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return NormalizedValue(value=deduped, data_type="list")
    
def normalize_string(value: str) -> NormalizedValue:
    cleaned = re.sub(r'\s+', ' ', value).strip()
    return NormalizedValue(value=cleaned, data_type="string")
    
def normalize_value(value: Any, kind: str) -> NormalizedValue:
    if kind == "identifier":
        return normalize_identifier(str(value))
    elif kind == "datetime":
        return normalize_date(str(value))
    elif kind == "contact":
        return normalize_contact(value)
    elif kind == "list":
        if not isinstance(value, list):
            value = [value]
        return normalize_list(value)
    else:
        return normalize_string(str(value))
