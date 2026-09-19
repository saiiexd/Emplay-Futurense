import os
import hashlib
from typing import Protocol
from src.config import CACHE_ROOT

class EmbeddingClient(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...
        
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

class OpenAIEmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            import logging
            logging.getLogger(__name__).warning("OPENAI_API_KEY not found in environment. OpenAI embeddings will fail.")
            
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            response = self.client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend([data.embedding for data in response.data])
        return all_embeddings

class FakeEmbeddingClient:
    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        import re
        words = re.sub(r'[^\w\s]', ' ', text.lower()).split()
        if not words:
            words = ["empty"]
            
        for word in words:
            base_hash = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            for i in range(self.dim):
                val = ((base_hash + i * 17) % 2000) / 1000.0 - 1.0
                vec[i] += val
            
        magnitude = sum(x**2 for x in vec) ** 0.5
        if magnitude == 0:
            return [0.0] * self.dim
        return [x / magnitude for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Fake client can do one by one, no network bottleneck
        return [self.embed_text(t) for t in texts]

class CachedEmbeddingClient:
    def __init__(self, client: EmbeddingClient, cache_dir: str = None):
        self.client = client
        self.cache_dir = cache_dir if cache_dir else os.path.join(CACHE_ROOT, ".embeddings_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, text: str) -> str:
        model_name = getattr(self.client, "model", "fake")
        key = f"{model_name}:{text}".encode('utf-8')
        hash_key = hashlib.sha256(key).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.json")

    def embed_text(self, text: str) -> list[float]:
        cache_path = self._get_cache_path(text)
        if os.path.exists(cache_path):
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        emb = self.client.embed_text(text)
        
        import json
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(emb, f)
            
        return emb

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = [None] * len(texts)
        uncached = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    results[i] = json.load(f)
            else:
                uncached.append(text)
                uncached_indices.append(i)
                
        if uncached:
            for i in range(0, len(uncached), 100):
                batch_texts = uncached[i:i+100]
                batch_indices = uncached_indices[i:i+100]
                batch_embs = self.client.embed_batch(batch_texts)
                for txt, idx, emb in zip(batch_texts, batch_indices, batch_embs):
                    results[idx] = emb
                    cache_path = self._get_cache_path(txt)
                    import json
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(emb, f)
        
        return results
