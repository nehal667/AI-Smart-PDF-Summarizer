import os
import tempfile
import streamlit as st
from processor import DocumentProcessor
from analyzer import LLMAnalyzer

st.set_page_config(page_title="Document Analyzer", layout="wide")
st.title("📄 AI Document Analyzer")

# Upload files
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, PPTX, or TXT files", 
    type=["pdf", "docx", "pptx", "txt"], 
    accept_multiple_files=True
)

# Choose mode
mode = st.radio("Choose summarization type:", ["single", "combined"])

# Run if files are uploaded
if uploaded_files:
    # Save files to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        # Initialize your processor and LLM analyzer
        processor = DocumentProcessor(temp_dir)
        analyzer = LLMAnalyzer(api_key=st.secrets["OPENROUTER_API_KEY"])

        if st.button("Generate Summary"):
            docs = processor.collect_texts()

            if mode == "single":
                st.header("📄 Individual Summaries")
                for fname, doc in docs.items():
                    st.subheader(f"**{fname}**")
                    summary = analyzer.analyze(doc)
                    st.text(summary)
            else:
                total_tokens, merged = processor.count_and_merge(docs)
                if total_tokens > processor.threshold:
                    st.error("⚠️ Combined document too long. Try 'single' instead.")
                else:
                    st.header("📚 Combined Summary")
                    summary = analyzer.analyze(merged.page_content)
                    st.text(summary)
