# Autonomous Data Analyst Agent

An agent that takes a CSV and a plain-English question, writes its own pandas
code to answer it, runs that code in a sandbox, self-corrects if it errors,
and explains the result in natural language — grounded by memory of earlier
questions in the same session (RAG) and optionally exposed as a real **MCP
server** that Claude Desktop can connect to.

## Architecture

```
CSV upload
   │
   ▼
Auto-EDA (core/auto_eda.py)         <- classic ML: schema profiling, IQR
   │                                    outlier detection, correlations
   ▼
User question ──► Memory retrieval (core/memory.py)   <- RAG: ChromaDB
   │                     │                                embeddings of
   │                     ▼                                past Q&A pairs
   │              similar past Q&A
   ▼                     │
LLM: text → pandas code ◄┘          <- NLP / GenAI: text-to-code generation
   │
   ▼
Sandboxed execution (core/tools.py)  <- Tool use: restricted exec()
   │
   ├─ error? ──► feed error back to LLM, retry (self-correction loop)
   │
   ▼
LLM: result → plain-English insight  <- GenAI: result interpretation
   │
   ▼
Store turn in memory (RAG write-back)
```

`mcp_server.py` exposes `load_csv`, `run_pandas_query`, and
`get_dataset_summary` as real MCP tools, so the same sandboxed pandas
executor can be driven from Claude Desktop directly — this is the piece
most fresher projects skip because it requires understanding the actual
protocol rather than just calling an LLM API.

## Setup

```bash
cd data-analyst-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your Anthropic API key
# get one at https://console.anthropic.com/settings/keys

streamlit run app.py
```

Open the local URL Streamlit prints, upload `sample_data/sales.csv` (or your
own CSV), and start asking questions like:

- "Which region has the highest total revenue?"
- "What's the average order value for returning vs new customers?"
- "Show me units sold by product category as a bar chart"

### Optional: run as an MCP server in Claude Desktop

```bash
pip install mcp
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "data-analyst": {
      "command": "python",
      "args": ["/absolute/path/to/data-analyst-agent/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop, start a new chat, and ask it to load your CSV and
analyze it — Claude will call your tools directly.

## What to build next (to make it deeper, not just wider)

Pick 2-3 of these to actually implement — don't just list them, ship them,
since "future work" bullets don't help in interviews but working extensions do:

1. **Persistent memory** — swap `chromadb.EphemeralClient()` for
   `PersistentClient()` so the agent remembers past sessions, not just the
   current one.
2. **Multi-file joins** — let the agent reason across two uploaded CSVs
   (e.g. orders + customers) by inferring join keys.
3. **Evaluation set** — write 20 question/expected-answer pairs and measure
   what % the agent gets right. This single addition is what turns "I built
   a cool demo" into "I evaluated my system," which is what interviewers
   actually want to hear.
4. **Guardrails** — add a check that rejects code trying to access
   `os`, `sys`, network calls, or file writes before execution, not just via
   restricted builtins (defense in depth).
5. **Deploy it** — Streamlit Community Cloud (free) so you can put a live
   demo link on your resume, not just a GitHub repo.

## Resume bullet points (edit numbers to match what you actually measured)

> Built an autonomous data-analyst agent that converts natural-language
> questions into pandas code, executes it in a sandboxed environment with
> self-correction on failure, and grounds explanations using session-level
> RAG memory (ChromaDB) for consistency across a conversation.

> Implemented a Model Context Protocol (MCP) server exposing the analysis
> pipeline as callable tools, enabling direct integration with Claude
> Desktop — demonstrating the emerging agent-tool interoperability standard.

> Designed an automatic EDA pipeline (IQR-based outlier detection,
> correlation analysis, schema profiling) that grounds LLM prompts in real
> dataset statistics, reducing hallucinated assumptions about the data.

## Talking points for interviews

- **Why sandboxed exec and not a safer alternative?** Be ready to discuss
  the tradeoff — restricted builtins vs. a full subprocess/container sandbox
  (gVisor, Docker) — and that you'd use the latter in production.
- **Why RAG here specifically?** It's not retrieving external documents,
  it's retrieving the agent's *own* past turns — explain why that's still
  RAG (embedding + retrieval + generation) even without an external corpus.
- **Why MCP instead of plain function calling?** MCP standardizes tool
  exposure across clients (Claude Desktop, other agents) instead of being
  wired to one app — that's the actual value prop, know it cold.
