import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

def load_file(path: str):
    """
    Load a single file (pdf, docx, txt, pptx) into a list of LangChain Documents.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    elif ext in (".docx", ".doc"):
        return UnstructuredWordDocumentLoader(path).load()
    elif ext == ".pptx":
        return UnstructuredPowerPointLoader(path).load()
    elif ext == ".txt":
        return TextLoader(path).load()
    else:
        print(f"Skipping unsupported file type: {ext}")
        return []
