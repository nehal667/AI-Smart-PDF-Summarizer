import os
from processor import DocumentProcessor
from analyzer import LLMAnalyzer

if __name__ == "__main__":
    folder = "pdfs"
    api_key = os.getenv("OPENROUTER_API_KEY", "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")  # your OpenRouter key

    proc = DocumentProcessor(folder)
    texts = proc.collect_texts()
    if not texts:
        print("No supported documents found.")
        exit()

    choice = input("Type 'single' for per-file summaries, or 'combined' for one summary: ").strip().lower()
    analyzer = LLMAnalyzer(api_key)

    if choice == "single":
        for fname, txt in texts.items():
            print(f"\n--- Summary for {fname} ---\n")
            print(analyzer.analyze(txt))
    else:
        total, merged_doc = proc.count_and_merge(texts)
        print(f"Total tokens: {total}")
        if total > proc.threshold:
            print("Token limit exceeded, cannot merge.")
        else:
            print("\n--- Combined Summary ---\n")
            print(analyzer.analyze(merged_doc.page_content))
