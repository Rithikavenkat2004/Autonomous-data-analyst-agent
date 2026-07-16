"""
core/agent.py
-------------
The orchestrator. This is the "agentic" part of the pipeline:

  user question
      -> retrieve similar past Q&A (RAG)
      -> LLM generates pandas code (NLP: text-to-code)
      -> tool executes code in sandbox (tool use)
      -> LLM turns raw result into a plain-English insight (GenAI)
      -> store this turn in memory for future consistency

Each step is a separate, inspectable function on purpose -- in an
interview you want to be able to point at exactly which step failed
and why, rather than a single black-box "agent.run()" call.
"""

import pandas as pd
from core.llm_client import LLMClient
from core.tools import run_pandas_code, get_schema_summary
from core.memory import QueryMemory


CODE_SYSTEM_PROMPT = """You are a senior data analyst. You write pandas code to answer
questions about a dataframe called `df`. Rules:
- Assign your final answer to a variable named `result`.
- `result` should be a DataFrame, Series, number, or string -- whatever best answers the question.
- Only use pandas (pd) and numpy (np), already imported. No file I/O, no imports, no exec/eval.
- Keep code short and correct. Do not include comments or explanations, only code.
- Do not wrap the code in markdown fences.
"""

INSIGHT_SYSTEM_PROMPT = """You are a data analyst explaining a result to a non-technical
stakeholder. Given the user's question and the raw computed result, write a 2-4 sentence
plain-English insight. Mention concrete numbers. Do not repeat the raw table verbatim,
interpret it. If the result suggests a follow-up question worth asking, mention it briefly.
"""


class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = LLMClient()
        self.memory = QueryMemory()
        self.schema = get_schema_summary(df)

    def _generate_code(self, question: str, similar_context: str) -> str:
        user_prompt = f"""Dataframe schema: {self.schema}

Relevant past questions and answers in this session (for consistency):
{similar_context}

New question: {question}

Write the pandas code."""
        return self.llm.generate(CODE_SYSTEM_PROMPT, user_prompt, max_tokens=400)

    def _generate_insight(self, question: str, result) -> str:
        result_str = str(result)[:2000]  # cap length for prompt safety
        user_prompt = f"Question: {question}\n\nComputed result:\n{result_str}"
        return self.llm.generate(INSIGHT_SYSTEM_PROMPT, user_prompt, max_tokens=300)

    def ask(self, question: str, max_retries: int = 2) -> dict:
        """
        Full pipeline for one user question. Returns a dict with the
        generated code, execution result, and natural-language insight,
        plus any error if all retries failed.
        """
        similar = self.memory.retrieve_similar(question)
        similar_context = self.memory.format_for_prompt(similar)

        code = self._generate_code(question, similar_context)
        exec_result = run_pandas_code(code, self.df)

        # Self-correction loop: if the generated code errors out, feed the
        # error back to the LLM and ask it to fix its own code.
        attempts = 0
        while not exec_result["success"] and attempts < max_retries:
            attempts += 1
            fix_prompt = f"""Dataframe schema: {self.schema}
Question: {question}
Your previous code:
{code}
It failed with this error:
{exec_result['error']}
Fix the code. Remember: assign the answer to `result`, no markdown fences."""
            code = self.llm.generate(CODE_SYSTEM_PROMPT, fix_prompt, max_tokens=400)
            exec_result = run_pandas_code(code, self.df)

        if not exec_result["success"]:
            return {
                "question": question,
                "code": code,
                "success": False,
                "error": exec_result["error"],
                "insight": "I couldn't compute an answer to this question after retrying. "
                           "Try rephrasing it or check if the referenced columns exist.",
            }

        insight = self._generate_insight(question, exec_result["result"])
        self.memory.add(question, code, insight)

        return {
            "question": question,
            "code": code,
            "success": True,
            "result": exec_result["result"],
            "insight": insight,
        }
