import os
import json
import logging
import re
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
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
            return "rfp_main"

    def parse_document(self, file_path: str) -> dict:
        """Parses a single PDF document and extracts its text page by page."""
        filename = os.path.basename(file_path)
        doc_type = self._infer_document_type(filename)
        
        parsed_data = {
            "metadata": {
                "source_filename": filename,
                "document_type": doc_type,
                "source_format": "pdf",
                "section_count": 0
            },
            "sections": []
        }

        try:
            doc = fitz.open(file_path)
            parsed_data["metadata"]["section_count"] = len(doc)
            
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract printed page label if possible (e.g. "Page 3 of 40")
                printed_page_label = None
                text_dict = page.get_text("dict")
                # Look at the first and last few blocks for headers/footers
                if "blocks" in text_dict:
                    blocks = text_dict["blocks"]
                    edge_blocks = blocks[:3] + blocks[-3:] if len(blocks) > 3 else blocks
                    for b in edge_blocks:
                        if b.get("type") == 0:  # text block
                            text = "".join([l["spans"][0]["text"] for l in b["lines"]]).strip()
                            if re.match(r'(?i)^(page\s+)?\d+\s+(of|\|)\s+\d+$', text):
                                printed_page_label = text
                                break

                # Extract standard text using blocks for better reading order
                blocks = page.get_text("blocks")
                text_blocks = []
                for b in blocks:
                    if b[6] == 0:  # Text block
                        text_blocks.append(b[4].strip())
                
                page_text = "\n\n".join(text_blocks)
                full_text += " " + page_text
                
                # Check for low text page (scanned or image only)
                words = page_text.split()
                low_text = len(words) < 10
                
                # Extract tables
                tables_data = []
                tables = page.find_tables()
                if tables and tables.tables:
                    for table in tables:
                        extracted = table.extract()
                        # clean up Nones
                        cleaned_table = [[str(cell) if cell is not None else "" for cell in row] for row in extracted]
                        tables_data.append(cleaned_table)
                
                section = {
                    "section_identifier": f"Page {page_num + 1}",
                    "text": page_text,
                    "metadata": {
                        "actual_page_index": page_num,
                        "low_text_page": low_text
                    }
                }
                if printed_page_label:
                    section["metadata"]["printed_page_label"] = printed_page_label
                if tables_data:
                    section["tables"] = tables_data
                    
                parsed_data["sections"].append(section)
            
            doc.close()
            
            # Extract addendum number from full text
            addendum_match = re.search(r'(?i)ADDENDUM\s+NO\.?\s*(\d+)', full_text)
            if addendum_match:
                try:
                    parsed_data["metadata"]["addendum_number"] = int(addendum_match.group(1))
                except ValueError:
                    pass
                
            # Extract doc date (simple regex for common formats)
            date_match = re.search(r'(?i)(?:date|issued):\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{4})', full_text)
            if date_match:
                parsed_data["metadata"]["doc_date"] = date_match.group(1)
                
            logger.info(f"Successfully parsed {filename} ({len(parsed_data['sections'])} sections).")
            return parsed_data

            
        except Exception as e:
            logger.error(f"Error parsing PDF {filename}: {e}")
            return None

    def process_directory(self, input_dir: str):
        """Processes all PDF files in the given directory."""
        if not os.path.exists(input_dir):
            logger.warning(f"Input directory {input_dir} does not exist.")
            return

        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.info(f"No PDF files found in {input_dir}.")
            return
            
        for filename in pdf_files:
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
