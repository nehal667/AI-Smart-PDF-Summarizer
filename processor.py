import os
import tiktoken
from langchain.docstore.document import Document
from loaders import load_file

class DocumentProcessor:
    def __init__(self, folder: str, token_threshold: int = 60000, model_name: str = "gpt-3.5-turbo"):
        self.folder = folder
        self.threshold = token_threshold
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def collect_texts(self):
        """Return dict mapping each filename to its full combined text."""
        texts = {}
        for fname in os.listdir(self.folder):
            path = os.path.join(self.folder, fname)
            docs = load_file(path)
            combined = " ".join(d.page_content for d in docs)
            if combined:
                texts[fname] = combined
        return texts

    def count_and_merge(self, texts: dict):
        """
        Count tokens across all texts and merge into one Document.
        Returns (total_tokens, merged_Document).
        """
        total_tokens = 0
        merged_parts = []
        for text in texts.values():
            total_tokens += len(self.tokenizer.encode(text))
            merged_parts.append(text)
        merged_doc = Document(page_content=" ".join(merged_parts))
        return total_tokens, merged_doc
