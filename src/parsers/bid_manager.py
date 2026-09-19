import os
import hashlib
import logging
from src.parsers.pdf_parser import PDFParser
from src.parsers.html_parser import HTMLParser

logger = logging.getLogger(__name__)

class BidManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pdf_parser = PDFParser(output_dir=os.path.join(output_dir, "pdf"))
        self.html_parser = HTMLParser(output_dir=os.path.join(output_dir, "html"))

    def process_bid_directory(self, bid_dir: str) -> dict:
        """Processes a single bid directory containing multiple files and returns a combined bid object."""
        bid_id = os.path.basename(os.path.normpath(bid_dir))
        bid_data = {
            "bid_id": bid_id,
            "documents": []
        }
        
        if not os.path.exists(bid_dir):
            logger.warning(f"Bid directory {bid_dir} does not exist.")
            return bid_data
            
            
        # Deterministic sorting
        filenames = sorted(os.listdir(bid_dir))
            
        for filename in filenames:
            file_path = os.path.join(bid_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            doc_id = hashlib.sha1(file_bytes).hexdigest()[:12]
                
            parsed = None
            if filename.lower().endswith('.pdf'):
                parsed = self.pdf_parser.parse_document(file_path)
            elif filename.lower().endswith(('.html', '.htm')):
                parsed = self.html_parser.parse_document(file_path)
                
            if parsed:
                parsed["metadata"]["doc_id"] = doc_id
                
                text_chars = 0
                low_text_pages = set()
                for sec in parsed.get("sections", []):
                    text_chars += len(sec.get("text", ""))
                    if sec.get("metadata", {}).get("low_text_page", False):
                        pn = sec.get("metadata", {}).get("page_number", 1)
                        low_text_pages.add(pn)
                        
                parsed["metadata"]["text_chars"] = text_chars
                parsed["metadata"]["low_text_pages"] = sorted(list(low_text_pages))
                
                bid_data["documents"].append(parsed)
                
        return bid_data
