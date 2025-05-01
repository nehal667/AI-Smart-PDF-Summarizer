from langchain_community.chat_models import ChatOpenAI

class LLMAnalyzer:
    def __init__(self, api_key: str, model: str = "openai/gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model=model,
            temperature=0.5
        )

    def analyze(self, text: str) -> str:
        prompt = (
            "Identify the type of this document, give a brief overview, "
            "and suggest section headings:\n\n"
            + text
        )
        resp = self.llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
