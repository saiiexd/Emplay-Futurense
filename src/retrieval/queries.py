from typing import List, Optional
from src.schemas.field_catalog import FIELD_CATALOG, FieldSpec
from src.schemas.enums import GroupName

def get_field_spec(field_name: str) -> Optional[FieldSpec]:
    return FIELD_CATALOG.get(field_name)

def get_fields_for_group(group_name: str) -> List[str]:
    """Returns a list of field names that belong to the specified group."""
    try:
        group_enum = GroupName(group_name)
    except ValueError:
        return []
    return [name for name, spec in FIELD_CATALOG.items() if spec.group == group_enum]

def get_aliases_for_group(group_name: str) -> List[str]:
    """Returns a combined list of BM25 aliases for all fields in a group."""
    aliases = []
    for field_name in get_fields_for_group(group_name):
        spec = get_field_spec(field_name)
        if spec:
            aliases.extend(spec.bm25_aliases)
    return list(set(aliases))  # deduplicate
