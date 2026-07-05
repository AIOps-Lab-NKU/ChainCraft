import os
import time
import random
import logging

from config import Config

os.environ["OPENAI_API_BASE"] = Config.OPENAI_API_BASE
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

from typing import List, Optional, Dict, Any
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

logger = logging.getLogger(__name__)

# Batching and retry configuration
_BATCH_SIZE = 5
_BATCH_DELAY = 1  # Delay between batches (seconds)
_MAX_RETRIES = 5
_BASE_DELAY = 2   # Retry base delay (seconds)


def _is_retryable(exc: Exception) -> bool:
    """Determine if exception is retryable (429 rate limit / 5xx server error)."""
    # OpenAI SDK APIError / RateLimitError etc. all have status_code attribute
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code == 429 or 500 <= status_code < 600
    # Fallback: check error message keywords
    msg = str(exc).lower()
    return ("rate limit" in msg or "429" in msg or "server error" in msg
            or "500" in msg or "502" in msg or "503" in msg or "504" in msg)


class StringInputEmbeddings(OpenAIEmbeddings):
   
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120)

        kwargs: Dict[str, Any] = {}
        kwargs["encoding_format"] = self.encoding_format
        kwargs["dimensions"] = self.dimensions

        # Batching
        batches = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
        total_batches = len(batches)
        all_embeddings: List[List[float]] = []

        for idx, batch in enumerate(batches, start=1):
            logger.info("Processing batch %d/%d, texts: %d", idx, total_batches, len(batch))

            # Exponential backoff retry
            succeeded = False
            last_exc: Optional[Exception] = None
            for attempt in range(_MAX_RETRIES + 1):  # 0 .. _MAX_RETRIES
                try:
                    resp = client.embeddings.create(
                        model=self.model,
                        input=batch,
                        **kwargs,
                    )
                    all_embeddings.extend(d.embedding for d in resp.data)
                    succeeded = True
                    break  # Success, exit retry
                except Exception as exc:
                    last_exc = exc
                    if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        break
                    delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "Batch %d/%d attempt %d/%d failed: %s. "
                        "Retrying in %.2fs ...",
                        idx, total_batches,
                        attempt + 1, _MAX_RETRIES,
                        exc, delay,
                    )
                    time.sleep(delay)

            # All retries exhausted or non-retryable, log error and raise
            if not succeeded:
                if last_exc is not None and not _is_retryable(last_exc):
                    logger.error(
                        "Batch %d/%d failed with non-retryable error: %s",
                        idx, total_batches, last_exc,
                    )
                else:
                    logger.error(
                        "Batch %d/%d failed after %d retries: %s",
                        idx, total_batches, _MAX_RETRIES, last_exc,
                    )
                raise last_exc

            # Delay between batches (no delay for last batch)
            if idx < total_batches:
                time.sleep(_BATCH_DELAY)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

if __name__ == "__main__":
    # Optional: enable HTTP debug logging

    emb = StringInputEmbeddings(
        model="text-embedding-v4",
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        encoding_format="float",
        dimensions=512,          
    )
    print(emb.embed_query("What is the capital of China?"))
