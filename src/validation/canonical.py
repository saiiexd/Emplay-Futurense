import re
from typing import Tuple, NamedTuple

class MatchResult(NamedTuple):
    matched: bool
    level: str
    original_substring: str
    start: int
    end: int

def standardize_punctuation(text: str) -> str:
    text = re.sub(r'[\u2018\u2019\u201B\u2032]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201F\u2033]', '"', text)
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    return text

def find_canonical_match(quote: str, text: str) -> MatchResult:
    if not quote or not text:
        return MatchResult(False, "none", "", -1, -1)
        
    # L1: Exact match
    idx = text.find(quote)
    if idx != -1:
        return MatchResult(True, "L1", text[idx:idx+len(quote)], idx, idx+len(quote))
        
    # L2 & L3: Token-based match with punctuation normalization
    def build_normalized_mapping(s: str) -> Tuple[str, list[int]]:
        res = []
        mapping = []
        s_std = standardize_punctuation(s)
        i = 0
        while i < len(s_std):
            if s_std[i].isspace():
                res.append(' ')
                mapping.append(i)
                while i + 1 < len(s_std) and s_std[i+1].isspace():
                    i += 1
            else:
                res.append(s_std[i])
                mapping.append(i)
            i += 1
        return "".join(res), mapping

    t_norm, t_norm_map = build_normalized_mapping(text)
    q_norm, _ = build_normalized_mapping(quote)
    
    idx_l2 = t_norm.find(q_norm)
    if idx_l2 != -1:
        start = t_norm_map[idx_l2]
        end = t_norm_map[idx_l2 + len(q_norm) - 1] + 1
        return MatchResult(True, "L2", text[start:end], start, end)
        
    idx_l3 = t_norm.lower().find(q_norm.lower())
    if idx_l3 != -1:
        start = t_norm_map[idx_l3]
        end = t_norm_map[idx_l3 + len(q_norm) - 1] + 1
        return MatchResult(True, "L3", text[start:end], start, end)
        
    # L4: Compact match (ignore all whitespace)
    def build_compact_mapping(s: str) -> Tuple[str, list[int]]:
        res = []
        mapping = []
        s_std = standardize_punctuation(s).lower()
        for i, char in enumerate(s_std):
            if not char.isspace():
                res.append(char)
                mapping.append(i)
        return "".join(res), mapping
        
    t_comp, t_comp_map = build_compact_mapping(text)
    q_comp, _ = build_compact_mapping(quote)
    
    idx_l4 = t_comp.find(q_comp)
    if idx_l4 != -1:
        start = t_comp_map[idx_l4]
        end = t_comp_map[idx_l4 + len(q_comp) - 1] + 1
        return MatchResult(True, "L4", text[start:end], start, end)
        
    return MatchResult(False, "none", "", -1, -1)
