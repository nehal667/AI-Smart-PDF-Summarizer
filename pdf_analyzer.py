import PyPDF2
import tiktoken
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

class PDFProcessor:
    def __init__(self, pdf_folder='pdfs', token_threshold=60000, model_name="gpt-3.5-turbo"):
        self.pdf_folder = pdf_folder
        self.token_threshold = token_threshold
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_pdf_tokens(self, pdf_paths):
        """Reads PDF files and counts the total number of tokens."""
        total_tokens = 0
        all_text = ""
        for pdf_path in pdf_paths:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                tokens = self.tokenizer.encode(text)
                total_tokens += len(tokens)
                all_text += text
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
        return total_tokens, all_text

    def load_pdfs_as_text(self, pdf_paths):
        """Reads PDF files and returns their content as a list of strings."""
        all_texts = []
        for pdf_path in pdf_paths:
            text = ""
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                all_texts.append(text)
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
        return all_texts

    def process_pdfs(self):
        """Processes the PDF files: counts tokens and returns a list of Langchain documents (one per PDF if within limit)."""
        pdf_files = [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        langchain_documents = []

        if not pdf_files:
            print(f"No PDF files found in the '{self.pdf_folder}' folder. Please add your PDFs there.")
            return langchain_documents

        total_tokens, all_text = self.count_pdf_tokens(pdf_files)
        print(f"Total tokens in all PDF files: {total_tokens}")

        if total_tokens < self.token_threshold:
            pdf_texts = self.load_pdfs_as_text(pdf_files)
            for i, text in enumerate(pdf_texts):
                langchain_documents.append(Document(page_content=text, metadata={"source": pdf_files[i]}))
            print(f"Loaded {len(langchain_documents)} PDFs as Langchain documents.")
        else:
            print(f"Total token count exceeds {self.token_threshold}. Processing PDFs individually.")
            pdf_texts = self.load_pdfs_as_text(pdf_files)
            for i, text in enumerate(pdf_texts):
                langchain_documents.append(Document(page_content=text, metadata={"source": pdf_files[i]}))
            print(f"Loaded {len(langchain_documents)} PDFs as Langchain documents (processed individually due to token limit).")

        return langchain_documents

class LLMAnalyzer:
    def __init__(self, api_key, model_name="google/gemini-pro", prompt_template="Identify the type of the following document and provide a brief overview. Also, list different sections that could be generated from this document based on its content:\n\n{document}"):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_template = prompt_template
        from langchain_community.llms.openrouter import OpenRouter
        self.llm = OpenRouter(api_key=self.api_key, model_name=self.model_name)

    def analyze_document(self, document_content, source):
        """Analyzes a single document content using the specified LLM via OpenRouter."""
        if not document_content:
            return f"No content to analyze for {source}."

        prompt = self.prompt_template.format(document=document_content)
        response = self.llm.invoke(prompt)
        return f"\n--- Summary for {source} ---\n\n{response}"

if __name__ == "__main__":
    # Configuration
    OPENROUTER_API_KEY = "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" # Replace with your actual API key

    # Create instances of the classes
    pdf_processor = PDFProcessor()
    llm_analyzer = LLMAnalyzer(api_key=OPENROUTER_API_KEY)

    # Process the PDFs
    documents = pdf_processor.process_pdfs()

    # Analyze each document
    if documents:
        print(f"\n--- Analyzing {len(documents)} documents ---")
        for doc in documents:
            analysis_result = llm_analyzer.analyze_document(doc.page_content, doc.metadata["source"])
            print(analysis_result)
    else:
        print("No documents were processed for analysis.")