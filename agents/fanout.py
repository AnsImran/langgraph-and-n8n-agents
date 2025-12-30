"""
Minimal LangGraph/LangChain prototype with a single OpenAI node.

Flow:
- parse free-form input (company/product/etc.)
- build a shared prompt
- call one OpenAI model (via OpenRouter) and show its output.

CLI examples:
  python -m agents.fanout --input-text "Company: ...\nProduct: ..." --run
  python -m agents.fanout --input-file path/to/block.txt --run
  python -m agents.fanout                # view parsed input + prompt only
"""

from __future__ import annotations

import argparse
import asyncio
import os
import textwrap
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

load_dotenv()
OPENROUTER_API_KEY = os.getenv("openrouter_api_key")

SYSTEM_PROMPT = (
    "You are a precise research assistant. Return concise bullet lists with real URLs whenever "
    "possible."
)


class GraphState(TypedDict, total=False):
    input_text: str
    meta: Dict[str, Any]
    prompt: str
    openai_output: str


def parse_input(raw: str) -> Dict[str, Any]:
    """Parse a loose text block into structured fields."""
    meta: Dict[str, Any] = {
        "company": None,
        "product": None,
        "product_type": None,
        "website": None,
        "notes": [],
    }
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("company:"):
            meta["company"] = line.split(":", 1)[1].strip() or None
        elif lower.startswith("product:"):
            meta["product"] = line.split(":", 1)[1].strip() or None
        elif lower.startswith("producttype:") or lower.startswith("type:"):
            meta["product_type"] = line.split(":", 1)[1].strip() or None
        elif lower.startswith("website:"):
            meta["website"] = line.split(":", 1)[1].strip() or None
        else:
            meta["notes"].append(line)
    return meta


def build_prompt(meta: Dict[str, Any]) -> str:
    """Construct the shared prompt for the model."""
    notes = "; ".join(meta["notes"]) if meta.get("notes") else "none"
    return textwrap.dedent(
        f"""
        You are a research scout helping collect ALL public information about a radiology product.
        Return two bullet lists: (1) downloadable assets (PDF, DOCX, TXT, brochures, manuals,
        white papers, spec sheets), and (2) web pages (product pages, integration docs, pricing,
        sales/demo links, regulatory/FDA info).

        For each item include: URL, the type of content, and 1 short reason it's relevant.
        Prefer U.S./California sources. If web search is unavailable, answer from your priors
        and typical locations for such material.

        Product: {meta.get('product') or 'unknown'}
        Company: {meta.get('company') or 'unknown'}
        Product type: {meta.get('product_type') or 'unknown'}
        Official site: {meta.get('website') or 'unknown'}
        Notes: {notes}
        """
    ).strip()


def _make_llm(model_id: str) -> ChatOpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing openrouter_api_key in environment or .env")
    return ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
        timeout=90,
        max_retries=2,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "DataCollectionScout",
        },
    )


# LangGraph nodes
async def node_parse(state: GraphState) -> GraphState:
    return {"meta": parse_input(state["input_text"])}


async def node_prompt(state: GraphState) -> GraphState:
    return {"prompt": build_prompt(state["meta"])}


def node_openai(model_id: str = "openai/gpt-4.1-mini"):
    llm = _make_llm(model_id)

    async def _node(state: GraphState) -> GraphState:
        try:
            resp = await llm.ainvoke([("system", SYSTEM_PROMPT), ("user", state["prompt"])])
            content = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # noqa: BLE001
            content = f"[error] {type(exc).__name__}: {exc}"
        return {"openai_output": content}

    return _node


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("parse", node_parse)
    graph.add_node("prompt", node_prompt)
    graph.add_node("openai", node_openai())

    graph.set_entry_point("parse")
    graph.add_edge("parse", "prompt")
    graph.add_edge("prompt", "openai")
    graph.add_edge("openai", END)
    return graph.compile()


GRAPH = build_graph()


async def run_single(raw_text: str) -> Dict[str, Any]:
    """Run the graph end-to-end and return state."""
    return await GRAPH.ainvoke({"input_text": raw_text})


def _read_input(args: argparse.Namespace) -> str:
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.input_text:
        return args.input_text
    return textwrap.dedent(
        """
        Company: Example Radiology Inc
        Product: RadSight AI
        ProductType: software
        Website: https://www.example.com/radsight
        Notes: focus on US radiology market; look for FDA/510(k) details
        """
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single OpenAI node via LangGraph.")
    parser.add_argument("--input-file", help="Path to a text file containing the product block.")
    parser.add_argument("--input-text", help="Inline text block for the product.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="If set, call the model (paid). If omitted, only show parsed input and prompt.",
    )
    args = parser.parse_args()

    raw = _read_input(args)
    meta = parse_input(raw)
    prompt = build_prompt(meta)
    print("Parsed input:")
    for k, v in meta.items():
        print(f"- {k}: {v}")
    print("\nPrompt to send:\n")
    print(prompt)

    if not args.run:
        print("\n(--run not set; skipping API call.)")
        return

    print("\nCalling OpenAI node (this may incur cost)...\n")
    state = asyncio.run(run_single(raw))
    output = state.get("openai_output", "")
    divider = "=" * 40
    print(f"\n{divider}\nOPENAI\n{divider}\n{output}\n")


if __name__ == "__main__":
    main()
