import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        """
        Initializes the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters when a section is split.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        if self.overlap >= self.max_chunk_size:
            raise ValueError("Overlap must be smaller than max_chunk_size")

    def _generate_chunk_id(self, source_filename: str, section_id: str, text: str) -> str:
        """Generates a deterministic unique identifier for a chunk based on its content."""
        raw_id = f"{source_filename}_{section_id}_{text}"
        return hashlib.md5(raw_id.encode('utf-8')).hexdigest()

    def _split_text(self, text: str) -> list[str]:
        """Splits a large text into smaller chunks by character with overlap."""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            
            # If we're not at the end and can find a space or newline nearby, try to break there to avoid splitting words
            if end < text_length:
                # Look backwards for a space or newline within the last 50 characters
                search_start = max(start + 1, end - 50)
                last_space = text.rfind(' ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                break_point = max(last_space, last_newline)
                if break_point != -1:
                    end = break_point + 1 # Include the space/newline
            
            chunks.append(text[start:end].strip())
            
            if end == text_length:
                break
                
            next_start = end - self.overlap
            if next_start <= start:
                # Fallback to prevent infinite loop if overlap is too close to max chunk size
                next_start = start + 1
            start = next_start
            
        return chunks

    def chunk_document(self, normalized_data: dict) -> list[dict]:
        """
        Takes a normalized document and divides it into meaningful chunks.
        
        Args:
            normalized_data: A dict conforming to the normalized schema.
            
        Returns:
            A list of chunk dictionaries.
        """
        if not normalized_data or "metadata" not in normalized_data or "sections" not in normalized_data:
            logger.warning("Invalid normalized data provided to chunker.")
            return []
            
        metadata = normalized_data["metadata"]
        source_filename = metadata.get("source_filename", "unknown")
        document_type = metadata.get("document_type", "unknown")
        source_format = metadata.get("source_format", "unknown")
        
        all_chunks = []
        
        current_combined_text = ""
        current_combined_ids = []
        
        def flush_combined():
            if current_combined_text:
                # Use the first section ID for the combined chunk, or join them
                combined_id = " & ".join(current_combined_ids)
                chunk_text = current_combined_text.strip()
                chunk_id = self._generate_chunk_id(source_filename, combined_id, chunk_text)
                
                chunk = {
                    "chunk_id": chunk_id,
                    "source_filename": source_filename,
                    "document_type": document_type,
                    "source_format": source_format,
                    "section_identifier": combined_id,
                    "text": chunk_text
                }
                
                # Add page information specifically if it's a PDF and section implies a page
                page_numbers = set()
                for cid in current_combined_ids:
                    if source_format == "pdf" and str(cid).lower().startswith("page"):
                        try:
                            page_num_str = str(cid).lower().replace("page", "").strip()
                            page_numbers.add(int(page_num_str))
                        except ValueError:
                            pass
                if page_numbers:
                    chunk["page_numbers"] = sorted(list(page_numbers))
                
                all_chunks.append(chunk)

        for section in normalized_data["sections"]:
            section_id = str(section.get("section_identifier", "unknown"))
            text = section.get("text", "").strip()
            
            if not text:
                continue
                
            # If the current section is large, or adding it would exceed the limit
            if len(text) > self.max_chunk_size:
                # Flush any existing combined small sections
                flush_combined()
                current_combined_text = ""
                current_combined_ids = []
                
                # Split large section
                text_splits = self._split_text(text)
                for i, split_text in enumerate(text_splits):
                    if not split_text:
                        continue
                        
                    chunk_id = self._generate_chunk_id(source_filename, section_id, split_text)
                    chunk = {
                        "chunk_id": chunk_id,
                        "source_filename": source_filename,
                        "document_type": document_type,
                        "source_format": source_format,
                        "section_identifier": section_id,
                        "text": split_text
                    }
                    if source_format == "pdf" and section_id.lower().startswith("page"):
                        try:
                            chunk["page_numbers"] = [int(section_id.lower().replace("page", "").strip())]
                        except ValueError:
                            pass
                    all_chunks.append(chunk)
            
            else:
                # It's a small section. Can we combine it with the current?
                separator = "\n\n" if current_combined_text else ""
                if len(current_combined_text) + len(separator) + len(text) <= self.max_chunk_size:
                    current_combined_text += separator + text
                    if section_id not in current_combined_ids:
                        current_combined_ids.append(section_id)
                else:
                    # Flush and start new combined block
                    flush_combined()
                    current_combined_text = text
                    current_combined_ids = [section_id]
                    
        # Flush any remaining combined text at the end
        flush_combined()
                
        logger.info(f"Chunked {source_filename} into {len(all_chunks)} chunks.")
        return all_chunks
