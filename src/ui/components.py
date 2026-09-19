import re

def highlight_text(text: str, query: str) -> str:
    """
    Highlights occurrences of `query` in `text` by wrapping them in HTML bold tags.
    Does not mutate the actual backend text string in memory, only returns a formatted copy.
    """
    if not query or not text:
        return text
    
    # Escape query to safely use in regex
    escaped_query = re.escape(query)
    
    # Wrap matches in a span with a background color or just bold
    # We use a markdown compatible HTML or just bold ** if we want to rely on Streamlit's markdown.
    # Streamlit's markdown supports some HTML, but standard markdown bold is safer.
    
    # We'll use markdown bolding: **match**
    # Since we want case-insensitive replace, we use a regex substitution.
    
    pattern = re.compile(f'({escaped_query})', re.IGNORECASE)
    highlighted = pattern.sub(r'**\1**', text)
    
    return highlighted
