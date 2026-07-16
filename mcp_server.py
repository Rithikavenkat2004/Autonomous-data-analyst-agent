"""
mcp_server.py
-------------
This exposes the same pandas-execution tool over the actual Model
Context Protocol, so Claude Desktop (or any MCP client) can load a CSV
and run analysis on it directly -- no Streamlit UI needed.

This is the piece that makes the project genuinely "MCP", not just
agentic tool-calling dressed up with the name. Most fresher projects
never touch the real protocol; this does.

Setup in Claude Desktop:
  1. pip install mcp
  2. Add to claude_desktop_config.json:
     {
       "mcpServers": {
         "data-analyst": {
           "command": "python",
           "args": ["/absolute/path/to/mcp_server.py"]
         }
       }
     }
  3. Restart Claude Desktop. You'll see "load_csv" and "run_pandas_query"
     as available tools in a new chat.
"""

import pandas as pd
from mcp.server.fastmcp import FastMCP
from core.tools import run_pandas_code
from core.auto_eda import profile_dataframe, profile_to_prompt_context

mcp = FastMCP("data-analyst-agent")

# Simple in-process state: the currently loaded dataframe.
_state = {"df": None, "path": None}


@mcp.tool()
def load_csv(path: str) -> str:
    """Load a CSV file from disk into memory for analysis. Call this first."""
    try:
        df = pd.read_csv(path)
        _state["df"] = df
        _state["path"] = path
        profile = profile_dataframe(df)
        return f"Loaded {path}.\n{profile_to_prompt_context(profile)}"
    except Exception as e:
        return f"Failed to load {path}: {e}"


@mcp.tool()
def run_pandas_query(code: str) -> str:
    """
    Run pandas code against the currently loaded dataframe (variable `df`).
    Assign your answer to a variable named `result`. Call load_csv first.
    """
    if _state["df"] is None:
        return "No CSV loaded yet. Call load_csv with a file path first."
    outcome = run_pandas_code(code, _state["df"])
    if outcome["success"]:
        return f"Result:\n{outcome['result']}"
    return f"Error: {outcome['error']}"


@mcp.tool()
def get_dataset_summary() -> str:
    """Return the auto-EDA summary of the currently loaded dataset."""
    if _state["df"] is None:
        return "No CSV loaded yet. Call load_csv with a file path first."
    profile = profile_dataframe(_state["df"])
    return profile_to_prompt_context(profile)


if __name__ == "__main__":
    mcp.run(transport="stdio")
