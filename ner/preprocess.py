"""Input preprocessing with bidirectional character-offset projection.

The runtime contract is char-offset spans into the *raw* user input. Any
cleaning step that changes length therefore has to remember which cleaned
char came from which original index, so predicted spans can be projected
back to the original coordinate system before they're returned.

Preprocessing is intentionally minimal and conservative for span-level NER:

  - NO casing changes        (PERSON/ORG capitalization is signal)
  - NO punctuation removal   (entity boundaries depend on commas, periods, |)
  - NO stop-word removal     (entities contain "of/in/&"; prepositions are
                              the role-position signal for PERSON vs ORG;
                              "no"/"not"/"without" are negation cues)
  - NO smart-quote / em-dash normalization (often part of entity surface)

What we DO clean by default:

  - zero-width chars     (ZWNJ/ZWJ/BOM/soft-hyphen — PDF/Excel paste noise)
  - control chars        (\x00-\x1f except \t \n \r)
  - whitespace chars     (NBSP / ideographic space / tabs / CR / VT / FF -> ' ')
  - whitespace runs      (collapse to single space, including inside entities)
  - leading/trailing ws  (strip)

NFC normalization is intentionally off by default: it can change length in
non-trivial ways for inputs with combining characters, and commercial English
text rarely benefits enough to be worth the position-map complexity.

The preprocessor is applied identically at training (after noise injection,
before JSONL write) and at inference (before tokenization). The
`PreprocessConfig` is serialized to `preprocess.json` in the artifact bundle
so train/inference can't drift.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

from ner.schema import Entity, Record


# Zero-width / format chars that commonly contaminate PDF/Word/Excel pastes.
_ZERO_WIDTH_CHARS: frozenset[str] = frozenset({
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte-order mark / zero-width no-break space
    "\u00ad",  # soft hyphen
})

# Unicode whitespace variants we fold to a plain space (length-preserving).
_WHITESPACE_TO_SPACE: dict[str, str] = {
    "\u00a0": " ",  # NBSP
    "\u2007": " ",  # figure space
    "\u202f": " ",  # narrow NBSP
    "\u3000": " ",  # ideographic space
    "\t": " ",
    "\r": " ",
    "\v": " ",
    "\f": " ",
}


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    strip_zero_width: bool = True
    strip_control_chars: bool = True
    normalize_whitespace_chars: bool = True
    collapse_whitespace: bool = True
    strip_leading_trailing: bool = True
    # NFC is opt-in — composing characters complicates the char map for
    # combining marks and rarely helps for English commercial text.
    nfc_normalize: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    """Result of cleaning one input.

    `char_map[i]` is the index in the *original* text of the cleaned char at
    position `i`. So `original_text[char_map[i]]` is the source character that
    produced `cleaned_text[i]` (or the first source character, if multiple
    were merged by whitespace collapse).
    """
    text: str
    char_map: tuple[int, ...]
    original_length: int

    def project_span(self, start: int, end: int) -> tuple[int, int]:
        """Project a span in cleaned coordinates back to original coordinates.

        Uses inclusive-start / exclusive-end semantics; the original span
        covers `original_text[result.char_map[start] : result.char_map[end-1] + 1]`,
        which always contains the surface text of the cleaned span.
        """
        if end <= start:
            raise ValueError(f"empty/invalid span [{start},{end})")
        if start < 0 or end > len(self.char_map):
            raise IndexError(
                f"span [{start},{end}) out of bounds for cleaned length "
                f"{len(self.char_map)}"
            )
        orig_start = self.char_map[start]
        orig_end = self.char_map[end - 1] + 1
        return orig_start, orig_end

    def project_entity(self, ent: Entity, original_text: str) -> Entity:
        """Project an Entity (in cleaned coordinates) to original coordinates.

        The returned Entity's `text` is taken from `original_text[orig_start:orig_end]`
        so `original_text[ent.start:ent.end] == ent.text` is preserved.
        """
        orig_start, orig_end = self.project_span(ent.start, ent.end)
        return Entity(
            type=ent.type,
            text=original_text[orig_start:orig_end],
            start=orig_start,
            end=orig_end,
            polarity=ent.polarity,
        )


class Preprocessor:
    """Apply `PreprocessConfig` to text or Records.

    Stateless besides `config`; safe to share across threads.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    # --- raw text cleaning -------------------------------------------------

    def clean(self, text: str) -> PreprocessResult:
        original_length = len(text)
        cfg = self.config

        # Phase 0: optional NFC over the *whole* string. NFC can compose
        # multiple source chars into one; we align the resulting chars to
        # the *first* source index of each composition. For commercial
        # English text this branch is effectively a no-op.
        if cfg.nfc_normalize:
            text, src_map = _nfc_with_source_map(text)
        else:
            src_map = list(range(len(text)))

        # Phase 1: per-character filters and substitutions.
        out_chars: list[str] = []
        char_map: list[int] = []
        for i, ch in enumerate(text):
            if cfg.strip_zero_width and ch in _ZERO_WIDTH_CHARS:
                continue
            if cfg.strip_control_chars and _is_strippable_control(ch):
                continue
            if cfg.normalize_whitespace_chars and ch in _WHITESPACE_TO_SPACE:
                out_chars.append(_WHITESPACE_TO_SPACE[ch])
            else:
                out_chars.append(ch)
            char_map.append(src_map[i])

        # Phase 2: collapse runs of whitespace to a single space. Keep the
        # source index of the *first* whitespace char in the run.
        if cfg.collapse_whitespace:
            collapsed_chars: list[str] = []
            collapsed_map: list[int] = []
            prev_is_ws = False
            for ch, src in zip(out_chars, char_map):
                if ch.isspace():
                    if prev_is_ws:
                        continue
                    collapsed_chars.append(" ")
                    collapsed_map.append(src)
                    prev_is_ws = True
                else:
                    collapsed_chars.append(ch)
                    collapsed_map.append(src)
                    prev_is_ws = False
            out_chars = collapsed_chars
            char_map = collapsed_map

        # Phase 3: trim leading / trailing whitespace.
        if cfg.strip_leading_trailing:
            while out_chars and out_chars[0].isspace():
                out_chars.pop(0)
                char_map.pop(0)
            while out_chars and out_chars[-1].isspace():
                out_chars.pop()
                char_map.pop()

        return PreprocessResult(
            text="".join(out_chars),
            char_map=tuple(char_map),
            original_length=original_length,
        )

    # --- training-side: project a Record into cleaned coordinates ----------

    def apply_to_record(self, record: Record) -> Record:
        """Return a new Record whose text/entities/preserve_spans are all in
        cleaned coordinates. Entities that get fully cleaned away are dropped.
        Validates the substring invariant on the result.
        """
        result = self.clean(record.text)
        if not result.text:
            # Pathological: everything cleaned away (e.g., all whitespace).
            # Return an empty record so callers can filter it out.
            return Record(text="", entities=[], meta=dict(record.meta or {}))

        # Inverse map: original_index -> cleaned_index (or -1 if dropped).
        inv = [-1] * result.original_length
        for clean_i, src_i in enumerate(result.char_map):
            # If multiple cleaned indices share a src (shouldn't happen in
            # this pipeline since collapse only keeps the first whitespace
            # of a run), the *first* assignment wins, which is what we want.
            if inv[src_i] == -1:
                inv[src_i] = clean_i

        def reproject(o_start: int, o_end: int) -> tuple[int, int] | None:
            new_start = -1
            for j in range(o_start, o_end):
                if inv[j] != -1:
                    new_start = inv[j]
                    break
            if new_start == -1:
                return None
            new_end = -1
            for j in range(o_end - 1, o_start - 1, -1):
                if inv[j] != -1:
                    new_end = inv[j] + 1
                    break
            if new_end <= new_start:
                return None
            return new_start, new_end

        new_entities: list[Entity] = []
        for ent in record.entities:
            rp = reproject(ent.start, ent.end)
            if rp is None:
                continue
            new_start, new_end = rp
            slice_text = result.text[new_start:new_end]
            if not slice_text:
                continue
            new_entities.append(Entity(
                type=ent.type, text=slice_text,
                start=new_start, end=new_end, polarity=ent.polarity,
            ))

        new_meta = dict(record.meta) if record.meta else {}
        preserve_spans = new_meta.get("preserve_spans")
        if preserve_spans:
            new_preserve: list[tuple[int, int]] = []
            for a, b in preserve_spans:
                rp = reproject(a, b)
                if rp is not None:
                    new_preserve.append(rp)
            if new_preserve:
                new_meta["preserve_spans"] = new_preserve
            else:
                del new_meta["preserve_spans"]

        out = Record(text=result.text, entities=new_entities, meta=new_meta)
        out.validate()
        return out

    def apply_to_records(self, records: Iterable[Record]) -> list[Record]:
        out: list[Record] = []
        for rec in records:
            cleaned = self.apply_to_record(rec)
            if cleaned.text:
                out.append(cleaned)
        return out

    # --- persistence (artifact bundle) -------------------------------------

    CONFIG_FILENAME: str = "preprocess.json"

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.config.to_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text())
        return cls(PreprocessConfig.from_dict(d))

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "Preprocessor":
        """Load `preprocess.json` from an artifact dir, falling back to
        default config if absent. Symmetric with how thresholds.json is
        loaded by the runtime."""
        path = Path(artifact_dir) / cls.CONFIG_FILENAME
        if path.exists():
            return cls.load(path)
        return cls(PreprocessConfig())

    def replace(self, **changes) -> "Preprocessor":
        """Return a new Preprocessor with overridden config fields."""
        return Preprocessor(replace(self.config, **changes))


# --- helpers -----------------------------------------------------------------


def _is_strippable_control(ch: str) -> bool:
    """True for control chars we want gone — NOT \\t \\n \\r."""
    cp = ord(ch)
    if cp == 0x7f:  # DEL
        return True
    if cp < 0x20 and ch not in ("\t", "\n", "\r"):
        return True
    return False


def _nfc_with_source_map(text: str) -> tuple[str, list[int]]:
    """NFC-normalize `text` and return (normalized_text, source_map).

    `source_map[i]` is the start index in the original of the (one or more)
    source chars that produced `normalized_text[i]`. When NFC composes
    multiple source chars into one, the composed char points at the first
    source char of the cluster.

    Implementation walks combining-character clusters: NFC composes within
    a base + combining marks group, so we group source chars by canonical
    combining class boundaries and align each composed-output char with the
    base of its source group.
    """
    import unicodedata

    # Group source chars into clusters: each cluster is one base char (ccc==0)
    # followed by any number of combining chars (ccc!=0).
    if not text:
        return "", []
    clusters: list[tuple[str, int]] = []
    cur_chars: list[str] = []
    cur_start = 0
    for i, ch in enumerate(text):
        if unicodedata.combining(ch) == 0 and cur_chars:
            clusters.append(("".join(cur_chars), cur_start))
            cur_chars = []
            cur_start = i
        cur_chars.append(ch)
    if cur_chars:
        clusters.append(("".join(cur_chars), cur_start))

    out_text_parts: list[str] = []
    out_map: list[int] = []
    for cluster, src_start in clusters:
        normalized = unicodedata.normalize("NFC", cluster)
        out_text_parts.append(normalized)
        # All output chars of this cluster map back to its first source char.
        # (Composition is the common case; if NFC happens to expand, we still
        # anchor every result to the cluster start — the overlap test in
        # apply_to_record handles boundary alignment.)
        out_map.extend([src_start] * len(normalized))
    return "".join(out_text_parts), out_map
