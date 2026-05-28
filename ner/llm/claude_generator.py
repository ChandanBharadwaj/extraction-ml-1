"""Optional Anthropic SDK hook for expanding seed pools and templates.

Two functions:
  - expand_entity_pool(entity_type, examples, n) -> list[str]
  - generate_free_text(prompt_examples, n) -> list[(text, [FreeGenEntity])]

Both use prompt caching on the long, repetitive system context so iterative
expansion runs stay cheap. The boundary rules from the TDD are encoded in the
system prompt verbatim.

This module is intentionally optional — the rest of the pipeline does not
import it. Install with `pip install -e .[data]`.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid hard import at runtime
    from anthropic import Anthropic

from ner.constants import ENTITY_TYPES
from ner.data.free_gen import FreeGenEntity

DEFAULT_MODEL = "claude-sonnet-4-6"

BOUNDARY_RULES = """\
Boundary rules (NEVER violate):
  COMMODITY: the raw material or product itself. EXCLUDE quantities, packaging,
             and units of measure. Example: capture "robust coffee", drop
             "500 tons of".
  ADDRESS:   the fullest available contiguous locational string, INCLUDING
             postal codes and countries.
  PERSON:    a human name; EXCLUDE titles ("Dr.", "Mr.") and job roles unless
             inextricably part of the name.
  ORG:       an organizational name; EXCLUDE generic legal-entity prefixes
             unless inextricably part of the name (keep "GmbH" / "Co." suffixes
             when present in the source).
Output format must be strict JSON; the caller will validate every span by
exact substring match.
"""


def _client() -> "Anthropic":
    from anthropic import Anthropic
    return Anthropic()


def expand_entity_pool(
    entity_type: str,
    examples: list[str],
    n: int,
    *,
    model: str = DEFAULT_MODEL,
    client: "Anthropic | None" = None,
) -> list[str]:
    """Ask Claude to extend a pool with `n` more diverse, on-policy values."""
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unknown entity type: {entity_type}")
    client = client or _client()

    system = [
        {
            "type": "text",
            "text": (
                "You generate seed pools for a Named Entity Recognition system.\n\n"
                + BOUNDARY_RULES
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user = (
        f"Generate {n} additional diverse {entity_type} surface forms, "
        "respecting the boundary rules. Return strict JSON:\n"
        '{"values": ["...", "..."]}\n\n'
        f"Existing examples:\n{json.dumps(examples)}"
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    obj = json.loads(text)
    return list(obj["values"])


def generate_free_text(
    n: int,
    *,
    model: str = DEFAULT_MODEL,
    client: "Anthropic | None" = None,
) -> list[tuple[str, list[FreeGenEntity]]]:
    """Generate `n` naturalistic records with declared entity surfaces.

    The Python caller is expected to run `relocate_entities` to compute the
    actual char offsets; we never trust the LLM to count characters.
    """
    client = client or _client()
    system = [
        {
            "type": "text",
            "text": (
                "You generate naturalistic short commercial text records for an "
                "NER training corpus.\n\n" + BOUNDARY_RULES + "\n"
                "Return strict JSON with the schema:\n"
                '{"records": [{"text": "...", '
                '"entities": [{"type": "PERSON|ORG|ADDRESS|COMMODITY", "text": "..."}]}]}'
                "\nEvery entity surface MUST appear verbatim somewhere in `text`."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user = (
        f"Generate {n} records mimicking invoices, manifests, emails, and supply "
        "chain notes. Vary length, formatting, and noise."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    obj = json.loads(text)
    out: list[tuple[str, list[FreeGenEntity]]] = []
    for rec in obj["records"]:
        ents = [FreeGenEntity(type=e["type"], text=e["text"]) for e in rec["entities"]]
        out.append((rec["text"], ents))
    return out
