"""
core/memory.py
---------------
This is the RAG layer. Every question the user asks (and the code/answer
generated for it) gets embedded and stored. When a new question comes in,
we retrieve the most similar past Q&A pairs and stuff them into the
prompt -- this keeps the agent's answers consistent within a session
(e.g. if it defined "high value customer" as spend > 1000 earlier, it
reuses that definition instead of contradicting itself later).

Implementation note: this uses TF-IDF vectors (scikit-learn) + cosine
similarity instead of a neural embedding model or vector database.
Earlier versions of this file tried sentence-transformers (torch-based)
and chromadb (needs a C++ compiler) -- both cause dependency headaches
on Windows (DLL load failures, missing MSVC build tools). TF-IDF has
zero native-compile dependencies and installs cleanly everywhere.

This is still genuinely RAG: vectorize -> store -> retrieve by cosine
similarity -> inject into the prompt. TF-IDF is a legitimate, classical
retrieval technique (it's what search engines used before neural
embeddings) -- in an interview you can honestly say you evaluated both
and chose TF-IDF for reliability, then mention neural embeddings
(sentence-transformers/OpenAI embeddings) as the natural upgrade path.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class QueryMemory:
    def __init__(self):
        self.records = []  # list of {"question": str, "code": str, "answer": str}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None  # TF-IDF matrix over all stored questions, rebuilt on add()

    def add(self, question: str, code: str, answer: str):
        self.records.append({"question": question, "code": code, "answer": answer})
        questions = [r["question"] for r in self.records]
        # Refit each time -- fine at session scale (dozens/hundreds of turns).
        self._matrix = self.vectorizer.fit_transform(questions)

    def retrieve_similar(self, question: str, k: int = 3) -> list[dict]:
        if not self.records:
            return []
        query_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [
            {"question": self.records[i]["question"], "code": self.records[i]["code"],
             "answer": self.records[i]["answer"]}
            for i in ranked[:k] if sims[i] > 0.1  # skip weakly related matches
        ]

    def format_for_prompt(self, similar: list[dict]) -> str:
        if not similar:
            return "No relevant past queries."
        lines = []
        for i, item in enumerate(similar, 1):
            lines.append(f"{i}. Q: {item['question']}\n   A: {item['answer']}")
        return "\n".join(lines)
