"""Production seed-data source modules.

Each submodule exports plain Python lists/dicts of *surface strings* (no SQL
escaping, no char offsets). `scripts.build_seed` imports them, deduplicates,
validates template/decoy-slot consistency, and emits the SQLite and Postgres
seed SQL files (`sql/seed.sql`, `sql/postgres/seed.sql`).

Contract for the slot-fill grammar (see ner/data/slot_fill.py):
  - Entity pool values are the *exact gold span text* a template inserts and
    labels. Titles / quantities / roles that must sit OUTSIDE the span belong
    in templates as literals or decoy slots, never in the entity value.
  - The single canonical set of decoy slot names is `ALLOWED_DECOY_SLOTS`.
    Templates may only reference these via `{decoy:<slot>}`.
"""
from __future__ import annotations

# The canonical decoy-slot vocabulary. `decoys.DECOYS` must provide a non-empty
# list for every name here, and `templates.TEMPLATES` may only reference these.
ALLOWED_DECOY_SLOTS: tuple[str, ...] = (
    "qty",            # quantity+unit prefixes, excluded from COMMODITY span ("500 tons of")
    "unit",           # bare units ("kg", "MT")
    "packaging",      # container nouns excluded from COMMODITY span ("bales of")
    "invoice_id",     # "PO#88231", "INV-2024-0421"
    "ref_id",         # "BL-77821", "REF#A-9912"
    "date",           # "2024-09-21", "21 Sept 2024 14:32 UTC"
    "currency",       # "$", "€", "₹"
    "role",           # job role excluded from PERSON span ("CEO", "buyer")
    "title",          # honorific excluded from PERSON span ("Mr.", "Dr.", "Shri")
    "salutation",     # "Dear", "Attn:"
    "signoff",        # "Best regards,", "Sent from my iPhone", "--"
    "boilerplate",    # full zero-entity sentences ("This is an automated message.")
    "neg_cue",        # PRESERVED: "does not contain", "no", "without" ...
    "contrast_cue",   # PRESERVED: ", but only", ", instead" ...
    "frozen_compound",# dissolved negation cues, NOT negations ("sugar-free")
    "incoterm",       # "FOB", "CIF", "DDP"
    # Cargo-document furniture: the top COMMODITY false-positive sources on
    # bills of lading / manifests. All excluded from every entity span.
    "vessel",         # vessel names ("MV EVER GIVEN", "M/T SEA PEARL V.023E")
    "container_spec", # container count/type ("1x20GP", "2X40HC")
    "port",           # port names / UN-LOCODEs ("CNSHA", "NHAVA SHEVA")
    "carrier",        # carrier SCAC codes ("MAEU", "MSCU") — full carrier
                      # company names belong in the ORG pool, not here
    "seal_marks",     # seal / marks-and-numbers lines ("SEAL NO. CN1234567")
    "qty_bare",       # bare qty+unit, no trailing "of" ("12,000 KG", "500 BAGS")
    "bulk_packaging", # stowage phrase after the span ("in 50kg PP bags")
    "dg_class",       # hazard metadata ("IMO CLASS 8", "PACKING GROUP II")
)
