"""
core/tools.py
-------------
These are the "tools" the agent can call. In a full MCP setup, each of
these functions is exposed over the MCP protocol (see mcp_server.py) so
that Claude Desktop or any MCP client can call them directly. Here they're
also wrapped in plain Python so the Streamlit app can call them locally
without needing a running MCP server.

SECURITY NOTE: exec() on LLM-generated code is inherently risky. This
sandbox restricts builtins and only exposes pandas/numpy/plotly -- good
enough for a portfolio project, NOT production-safe. Mention this
tradeoff explicitly in interviews; it shows security awareness.
"""

import io
import contextlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio


SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "round": round, "abs": abs,
    "str": str, "int": int, "float": float, "bool": bool, "print": print,
}


def run_pandas_code(code: str, df: pd.DataFrame) -> dict:
    """
    Executes LLM-generated pandas code in a restricted namespace.
    The code MUST assign its final answer to a variable called `result`.
    Returns {"success": bool, "result": ..., "stdout": str, "error": str}
    """
    local_ns = {"df": df.copy(), "pd": pd, "np": np, "px": px}
    global_ns = {"__builtins__": SAFE_BUILTINS}

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, global_ns, local_ns)
        result = local_ns.get("result", None)

        # Make result JSON/Streamlit friendly
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result = result.head(50) if isinstance(result, pd.DataFrame) else result.head(50)

        return {
            "success": True,
            "result": result,
            "stdout": stdout_capture.getvalue(),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "stdout": stdout_capture.getvalue(),
            "error": f"{type(e).__name__}: {e}",
        }


def generate_chart(chart_type: str, df: pd.DataFrame, x: str, y: str = None, color: str = None):
    """Generates a Plotly chart. Returns a plotly Figure or None on failure."""
    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x, color=color)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y)
        else:
            return None
        fig.update_layout(template="plotly_white")
        return fig
    except Exception:
        return None


def get_schema_summary(df: pd.DataFrame) -> str:
    """Compact schema string for prompting."""
    return ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
