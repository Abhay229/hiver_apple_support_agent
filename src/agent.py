"""
Core agent: given an incoming customer message, classify intent, decide
AUTO_HANDLE vs ESCALATE, and draft a reply grounded in real historical
AppleSupport resolutions retrieved for this specific message.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import json
import re

from intents import INTENTS, ROUTING_POLICY, INTENT_LIST
from retrieval import HistoricalRetriever
from llm_client import complete, get_model


def _build_system_prompt() -> str:
    intent_block = "\n".join(
        f"- {name}: {info['description']}" for name, info in INTENTS.items()
    )
    return f"""You are a support-triage assistant for AppleSupport on Twitter.

Classify each incoming customer message into exactly one of these intents:
{intent_block}

{ROUTING_POLICY}

When drafting a reply:
- Ground it in the style and content of the real historical AppleSupport
  replies you are given as examples below -- match their tone (brief, warm,
  helpful) and their actual resolution pattern.
- Do NOT invent URLs, article names, or specific instructions that don't
  appear in the retrieved historical examples. If none of the retrieved
  examples contain a public fix, ask a grounded clarifying question instead
  of fabricating one.
- Keep it Twitter-reply length (under 280 characters).

Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "intent": "<one of: {', '.join(INTENT_LIST)}>",
  "intent_confidence": "<low|medium|high>",
  "routing": "<AUTO_HANDLE|ESCALATE>",
  "routing_reason": "<one sentence, tied to the stated policy>",
  "draft_reply": "<the drafted reply text>"
}}
"""


def _build_user_prompt(customer_message: str, retrieved: list[dict]) -> str:
    examples_block = "\n\n".join(
        f"Past customer message: {r['customer_text']}\n"
        f"AppleSupport's actual reply: {r['reply_text']}"
        for r in retrieved
    ) or "(no closely similar historical example found)"

    return f"""New incoming customer message:
"{customer_message}"

Similar past resolved conversations (for grounding your reply, most similar first):
{examples_block}

Classify this message and draft a reply."""


def _parse_json_response(raw_text: str) -> dict:
    # Strip accidental code fences defensively, even though the prompt forbids them.
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


class SupportAgent:
    def __init__(self, retriever: HistoricalRetriever, model: str | None = None):
        self.retriever = retriever
        self.model = model or get_model("AGENT_MODEL")
        self.system_prompt = _build_system_prompt()

    def handle(self, customer_message: str, k_retrieved: int = 3) -> dict:
        retrieved = self.retriever.retrieve(customer_message, k=k_retrieved)
        user_prompt = _build_user_prompt(customer_message, retrieved)

        raw_text = complete(self.system_prompt, user_prompt, self.model, max_tokens=500)
        try:
            parsed = _parse_json_response(raw_text)
        except json.JSONDecodeError:
            # Fail loudly with the raw text attached rather than silently
            # returning garbage -- easier to debug at eval time.
            parsed = {
                "intent": "PARSE_ERROR",
                "intent_confidence": "low",
                "routing": "ESCALATE",
                "routing_reason": "Model output failed to parse as JSON",
                "draft_reply": "",
                "_raw": raw_text,
            }
        parsed["_retrieved_examples"] = retrieved
        return parsed
