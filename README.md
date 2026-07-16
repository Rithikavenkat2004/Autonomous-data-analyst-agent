# Autonomous Data Analyst Agent

Upload a CSV, ask questions in plain English. The agent writes its own
pandas code, runs it in a sandbox, self-corrects on errors, and explains
results in natural language — using session memory (RAG) for consistency
across questions. Also ships a real MCP server for Claude Desktop.

## Concepts used
- **ML** — automatic EDA: IQR outlier detection, correlation analysis
- **NLP/GenAI** — text-to-pandas-code generation, result-to-English explanation
- **RAG** — TF-IDF + cosine similarity retrieval of past Q&A for consistency
- **MCP** — real Model Context Protocol server (`mcp_server.py`), connectable to Claude Desktop
- **Agentic behavior** — self-correction loop that retries when generated code errors

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your free OpenRouter API key (no card needed): https://openrouter.ai
streamlit run app.py
```

Upload `sample_data/sales.csv`, then ask things like:
- "Which product category has the highest total revenue?"
- "Compare average order value between new and returning customers"
- "Show units sold by category as a bar chart"

## Architecture

```
CSV upload -> Auto-EDA (ML: outliers, correlations)
           -> Question -> retrieve similar past Q&A (RAG)
           -> LLM generates pandas code (NLP/GenAI)
           -> Sandboxed execution -> error? retry with fix (self-correction)
           -> LLM explains result in plain English
           -> Store turn in memory for next question
```

## Resume bullet

> Built an autonomous data-analyst agent that converts natural-language
> questions into pandas code, executes it in a sandboxed environment with
> self-correction on failure, and grounds explanations using RAG-based
> session memory. Also implemented a Model Context Protocol (MCP) server
> exposing the pipeline as tools for Claude Desktop.

## Interview talking points

- **Sandboxed exec** — restricted builtins, not production-safe; would use
  a container sandbox (Docker/gVisor) for real deployment.
- **RAG here** — retrieves the agent's own past turns (not an external
  corpus), still genuinely RAG: embed, store, retrieve, inject into prompt.
- **MCP vs plain function calling** — MCP standardizes tool exposure across
  clients instead of wiring tools to one specific app.
