import os
import json
import logging
import re
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HTMLParser:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _infer_document_type(self, filename: str) -> str:
        """Infer the document type based on the filename."""
        lower_name = filename.lower()
        if "addendum" in lower_name:
            return "addendum"
        elif "affidavit" in lower_name:
            return "affidavit"
        elif "spec" in lower_name:
            return "specification"
        else:
            return "portal_listing"

    def parse_document(self, file_path: str) -> dict:
        """Parses a single HTML document and extracts its text."""
        filename = os.path.basename(file_path)
        doc_type = self._infer_document_type(filename)
        
        parsed_data = {
            "metadata": {
                "source_filename": filename,
                "document_type": doc_type,
                "source_format": "html",
                "section_count": 0
            },
            "sections": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
            # Remove irrelevant elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link", "noscript"]):
                element.decompose()
                
            # Extract meaningful text and structured sections
            current_section = "Main Content"
            current_text = []
            
            def flush_section():
                if current_text:
                    t = " ".join(current_text).strip()
                    if t:
                        if re.search(r'See more(\s+description)?', t, re.IGNORECASE):
                            parsed_data["metadata"]["truncated_description"] = True
                            t = re.sub(r'See more(\s+description)?', '', t, flags=re.IGNORECASE).strip()
                        if t:
                            parsed_data["sections"].append({
                                "section_identifier": current_section,
                                "text": t
                            })
                    current_text.clear()

            full_text = soup.get_text(separator=' ', strip=True)
            
            # Extract addendum number from full text
            addendum_match = re.search(r'(?i)ADDENDUM\s+NO\.?\s*(\d+)', full_text)
            if addendum_match:
                parsed_data["metadata"]["addendum_number"] = int(addendum_match.group(1))
            else:
                parsed_data["metadata"]["addendum_number"] = None
                
            # Extract doc date (simple regex for common formats)
            date_match = re.search(r'(?i)(?:date|issued):\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{4})', full_text)
            if date_match:
                parsed_data["metadata"]["doc_date"] = date_match.group(1)

            if soup.body:
                for element in soup.body.descendants:
                    # Ignore elements that don't have strings or are NavigableString directly
                    # BeautifulSoup returns NavigableStrings directly in descendants
                    from bs4 import NavigableString, Comment, Doctype, CData, ProcessingInstruction
                    if isinstance(element, (Comment, Doctype, CData, ProcessingInstruction)):
                        continue
                    if isinstance(element, NavigableString):
                        text = str(element).strip()
                        if not text:
                            continue
                            
                        parent = element.parent
                        is_label = False
                        # Check if it looks like a label
                        if parent.name in ['strong', 'b', 'th', 'label'] and text.endswith(':'):
                            is_label = True
                        elif text.endswith(':') and len(text) < 50:
                            is_label = True
                            
                        if is_label:
                            flush_section()
                            current_section = text.rstrip(':')
                            current_text.append(text)
                        else:
                            current_text.append(text)
                flush_section()
            else:
                if full_text:
                    if re.search(r'See more(\s+description)?', full_text, re.IGNORECASE):
                        parsed_data["metadata"]["truncated_description"] = True
                        full_text = re.sub(r'See more(\s+description)?', '', full_text, flags=re.IGNORECASE).strip()
                    parsed_data["sections"].append({
                        "section_identifier": "Main Content",
                        "text": full_text
                    })

            parsed_data["metadata"]["section_count"] = len(parsed_data["sections"])
                
            logger.info(f"Successfully parsed {filename} ({len(parsed_data['sections'])} sections).")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing HTML {filename}: {e}")
            return None

    def process_directory(self, input_dir: str):
        """Processes all HTML files in the given directory."""
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory {input_dir} does not exist.")
            return

        html_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.html', '.htm'))]
        
        if not html_files:
            logger.info(f"No HTML files found in {input_dir}.")
            return
            
        for filename in html_files:
            file_path = os.path.join(input_dir, filename)
            logger.info(f"Processing {filename}...")
            
            parsed_data = self.parse_document(file_path)
            if parsed_data:
                # Save to processed data area
                output_filename = f"{os.path.splitext(filename)[0]}_parsed.json"
                output_path = os.path.join(self.output_dir, output_filename)
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved parsed data to {output_filename}")
                except Exception as e:
                    logger.error(f"Error saving output for {filename}: {e}")
