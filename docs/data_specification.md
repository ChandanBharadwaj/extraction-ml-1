# Data Specification — Multi-Entity NER

This document is the contract for the data effort. It specifies what training,
validation, and test data must look like for the system to reach production
quality. Anything not enumerated here is out of scope; anything missing must
be added before signing off on a release.

The model extracts four entity types — `PERSON`, `ORG`, `ADDRESS`,
`COMMODITY` — with `polarity ∈ {POS, NEG}` on `COMMODITY`. Outputs are
half-open char spans `[start, end)` into the raw input.

---

## 1. Three datasets — definitions and contracts

| Set | Source | Size target | Role | Refresh cadence |
|---|---|---|---|---|
| **Train** | 100% synthetic (slot-fill + LLM free-gen + noise + preprocess) | 50k–500k records | Loss / gradient updates only | Regenerated whenever pools, templates, or `preprocess.json` change |
| **Dev** | 100% real, hand-labeled, character offsets verified by Python | 500–2,000 records | Early stopping, threshold tuning, hyperparameter selection, error analysis | Quarterly, with active-learning additions |
| **Test** | 100% real, hand-labeled, **disjoint from dev**, locked vault | 500–1,000 records | Release sign-off only. Touched once per release candidate. | Annually, frozen between refreshes |

Contracts that cannot be broken:

- **Train never touches dev or test.** Synthetic records are tagged with their
  template id in `meta`; if any real-text-derived record makes it into the
  train pool the run is invalid.
- **Dev and test are disjoint.** Hashes of `text` checked before either set
  is finalized. Records with `text` similarity (token Jaccard) > 0.9 between
  the two sets are split (one moves to one side; never both).
- **Test is locked.** Touching test data more than once per release candidate
  is data leakage; the test number stops being meaningful.
- **All gold offsets verified by Python.** Every record must satisfy
  `text[start:end] == ent.text` for every entity at load time. Annotators
  never write offsets; they highlight spans and a tool computes offsets.

---

## 2. Size targets and minimum support

### 2.1 Train

- Volume: 50k–500k records.
- Per template: at least 200 instances after sampling, to ensure each
  syntactic pattern is well-represented.
- Per entity value: each commodity / org / address / person value in the
  pool should appear in at least 5 records across templates.

### 2.2 Dev

Minimum support per entity bucket (records where the bucket appears at
least once):

| Bucket | Min records |
|---|---|
| `PERSON` | 200 |
| `ORG` | 250 |
| `ADDRESS` | 200 |
| `COMMODITY (POS)` | 300 |
| `COMMODITY (NEG)` | 150 |
| Records with **zero** entities (negatives) | 100 |
| Frozen-compound positives (`sugar-free`, `non-stick`, `stainless`) | 50 |
| Adversarial / ambiguous (Tesla, Ford, Hermès, eponymous brands) | 75 |

Total dev set ≥ 500 records, target 1,000.

### 2.3 Test

Same bucket structure as dev, half the minimums, target 750 records total.
Test is built from a **later time window** than dev whenever possible to
catch temporal drift early.

---

## 3. Span length distribution (per entity type)

The model has to handle short single-token entities and long multi-word
phrases. Each length bucket must be represented in every split.

### 3.1 Length buckets (character count)

| Bucket | Char range | Token range (rough) |
|---|---|---|
| **XS** | 1–3 | 1 |
| **S** | 4–10 | 1–2 |
| **M** | 11–25 | 2–4 |
| **L** | 26–60 | 4–10 |
| **XL** | 61–120 | 10–20 |
| **XXL** | 121+ | 20+ |

### 3.2 Minimum support per (type, bucket) in dev

The system fails differently at the edges (XS = ambiguity with common
nouns; XXL = address truncation). Every cell must be ≥ 10 records;
the marked cells are higher-priority because they're typical:

| Type \ Bucket | XS | S | M | L | XL | XXL |
|---|---|---|---|---|---|---|
| `PERSON` | 10 (≥1 token) | **40** | **80** | 25 | 10 | 0 (rare) |
| `ORG` | 10 (acronyms) | **50** | **80** | 50 | 25 | 15 |
| `ADDRESS` | 0 (impossible) | 10 (PO boxes) | 30 | **80** | **70** | 30 |
| `COMMODITY` (POS+NEG) | 25 (e.g. "oil") | **75** | **100** | 60 | 30 | 10 (long product codes) |

Buckets marked bold are the high-frequency real-world cases; the rest are
tail-of-distribution but still required.

### 3.3 Word-count distribution (informational, not enforced)

- `PERSON`: 1–4 words, mode 2 (first + last).
- `ORG`: 1–6 words, mode 2–3.
- `ADDRESS`: 3–15 words, mode 6.
- `COMMODITY`: 1–6 words, mode 2 with a long tail.

---

## 4. Per-entity-type coverage requirements

The dev and test sets must each cover every row below at least once. Train
should over-sample these patterns (templates calibrated to weight).

### 4.1 PERSON

- Single-name vs. first + last vs. first + middle + last.
- Mixed-script / non-Latin names (Mandarin, Devanagari, Cyrillic, Arabic).
- Hyphenated names (Mary-Kate, Smith-Jones).
- Names with apostrophes (O'Brien, D'Souza).
- Names with particles (van der Berg, de la Cruz, von Neumann).
- All-lowercase (OCR'd, fast-typed): `maria gonzalez`.
- ALL-CAPS: `MARIA GONZALEZ`.
- Titled but title **excluded** from span: `Mr. Smith` → `Smith`.
- Names that double as common nouns: `Mark`, `Will`, `Hope`.
- Names that double as ORG surface forms: `Tesla`, `Ford`, `Hermès`.
- Names with initials: `J. K. Rowling`, `R H Macy`.
- Suffixes (`Jr.`, `III`, `PhD`) included or excluded — pick one rule and
  document; default: included only if no separating comma (`John Smith Jr.`
  yes; `John Smith, Jr.` no).

### 4.2 ORG

- With explicit legal suffix: `Co.`, `Corp.`, `Ltd`, `LLC`, `Inc.`, `GmbH`,
  `S.A.`, `Pvt. Ltd`, `Pte Ltd`, `K.K.`, `OY`, `AG`, `BV`, `SAS`, `S.A. de C.V.`.
- Without suffix: `Acme Trading`, `Oceanic Freight`.
- Acronyms: `IBM`, `BHP`, `BASF`, `KPMG`.
- Brand-style names: `Apple`, `Tesla`, `Patagonia`, `Hermès`.
- With ampersand: `Procter & Gamble`, `Johnson & Johnson`.
- With slash / dash: `Marsh & McLennan`, `Bristol-Myers Squibb`.
- Multi-word descriptive: `International Business Machines Corporation`.
- Joint ventures: `Sumitomo Mitsui Banking Corporation`.
- Compound with location: `Bank of America`, `Standard Chartered Singapore`.
- Pipe-delimited tabular context: `| Acme Trading Co. |`.
- All-caps: `ACME TRADING CO.`.
- Lowercased / sloppy casing: `acme trading co`.

### 4.3 ADDRESS

- Single-line: `7 Canal St, Singapore 049320`.
- Multi-line (newlines or commas): `42 Industrial Park Road\nRotterdam\n3011 AB`.
- PO Box: `PO Box 1234, Sydney NSW 2000`.
- With unit / suite / floor: `Suite 5, 88 Wallaby Way, ...`.
- With country: include the country token in the span if present.
- US format: `1234 Main St, Springfield, IL 62701`.
- UK format: `10 Downing Street, London SW1A 2AA, United Kingdom`.
- Asian formats (postal code position varies): `Singapore 049320`,
  `Tokyo 100-0001, Japan`.
- With non-ASCII chars (umlauts, accented chars): `Köln`, `São Paulo`.
- Industrial park / warehouse style: `Plot 42, Sector 5, Gurugram, 122001`.
- Truncated (OCR cut off mid-line) — verify the partial span still validates.
- ALL-CAPS addresses.
- Mixed casing.

### 4.4 COMMODITY (without polarity, see Section 5 for negation)

- Bare common nouns: `coffee`, `steel`, `sugar`, `wood`.
- Qualified by grade: `Grade A robusta coffee`, `304 stainless steel sheet`.
- Qualified by treatment: `refined copper cathode`, `galvanized steel coil`,
  `anhydrous ammonia`, `raw cane sugar`.
- Branded as part of commodity name: `Polyethylene resin HDPE`,
  `LDPE 1810D` (alphanumeric product codes).
- Plurals vs. singular: `bales`, `containers`, `barrels` — the **container
  word is NOT part of the commodity span** (TDD rule).
- Quantity-prefixed: `500 tons of robusta coffee` — span over `robusta coffee`,
  exclude the quantity (TDD boundary rule).
- Adjacency: serial list `galvanized steel coil, anhydrous ammonia, raw cane
  sugar` — three distinct non-overlapping spans.
- Compound noun with frozen "negation-looking" cue: `sugar-free chocolate`,
  `stainless steel sheet`, `non-stick pan` — span includes the whole
  qualifier; **frozen cue must NOT trigger polarity=NEG**.
- Color / variety modifiers: `red wine`, `arabica coffee`, `organic cotton`.
- Industry-specific jargon: `Brent crude`, `LME copper`, `LNG`.

---

## 5. Negation coverage (COMMODITY only)

Polarity is the biggest correctness risk for COMMODITY. The four
linguistic properties from the design discussion translate into mandatory
data patterns.

### 5.1 Polarity buckets that must exist in every split

| Bucket | Description | Min records (dev) |
|---|---|---|
| **POS-only** | Asserted commodities; no negation cue in record | 200 |
| **NEG-only** | Single denied commodity, no other commodity in record | 50 |
| **Mixed POS+NEG** | Same record contains both polarities | 50 |
| **Multi-target NEG** | "does not contain X, Y, or Z" — 2+ NEG in one cue scope | 30 |
| **Qualified NEG + bare POS** | "no special wood, but ordinary wood is fine" | 20 |
| **Frozen-compound positive** | "sugar-free chocolate" / "non-stick pan" / "stainless steel" — surface "negation-looking" cue is **dissolved** and span is POS | 50 |
| **Word-order A** | denial then assertion ("no X, only Y") | 25 |
| **Word-order B** | assertion then denial ("X, but not Y") | 25 |
| **Negation cue diversity** | Cues used: `no`, `not`, `does not contain`, `without`, `lacks`, `excludes`, `free of`, `absent of`, `missing`, `cannot include` — each must appear ≥ 5 times | 50 (aggregate) |

### 5.2 Negation cue inventory

Real negation cues (trigger polarity=NEG when in scope):

- `no`, `not`, `none`, `never`
- `does not contain`, `do not contain`, `cannot contain`
- `without`, `lacks`, `lacking`, `excludes`, `excluding`
- `free of`, `absent of`, `missing`, `omitted`
- `unauthorized`, `forbidden`, `prohibited` (domain-specific; flag)

Frozen compounds (do **not** trigger negation):

- `sugar-free`, `gluten-free`, `lead-free`, `BPA-free`, `oil-free`,
  `fat-free`, `dairy-free`, `caffeine-free`, `tax-free`, `duty-free`
- `non-stick`, `non-toxic`, `non-flammable`, `non-allergenic`
- `stainless` (historically "without stain"), `seamless`, `careless`
- `noise-cancelling`, `wrinkle-resistant`

Contrast / scope cues (mark scope boundaries):

- `, but only`, `, only`, `, instead`, `, rather than`, `; instead of`,
  `; in lieu of`, `, except`, `, excluding`

### 5.3 Scope distance

The denied commodity span should appear at varying distances from its
cue, measured in tokens between cue end and span start:

| Distance | Examples | Min records (dev) |
|---|---|---|
| 0–1 tokens (immediately after cue) | "no wood" | 30 |
| 2–4 tokens | "does not contain any wood" | 30 |
| 5–10 tokens | "the shipment does not, per the inspection, contain wood" | 15 |
| Multi-target (cue applies to several spans) | "without wood, cotton, or steel" | 30 |

### 5.4 Subtype-qualifier discipline

When the cue applies to a qualified form, the NEG span must cover the
**full qualified phrase**, not just the bare commodity:

- "no special wood" → NEG span over `special wood` (not just `wood`).
- "no Grade A robusta coffee" → NEG span over `Grade A robusta coffee`.
- "no 304 stainless steel sheet" → NEG span over `304 stainless steel sheet`.

The bare commodity remaining POS in the same record (when present) is
labeled separately:

- "no special wood, but ordinary wood is fine" →
  NEG `special wood`, POS `wood` (or `ordinary wood` if "ordinary" is in
  the pool).

---

## 6. Record-level composition

### 6.1 Number-of-entities distribution

Every split must contain records across this distribution:

| Entities per record | % of split (target) | Notes |
|---|---|---|
| 0 (true negative) | 5–10% | Records that look like they have entities but don't (or only ambiguous noise). |
| 1 | 20% | Single-entity records. |
| 2 | 25% | Most common in invoice headers. |
| 3 | 25% | Typical manifest density. |
| 4 | 15% | Full invoice header + commodity line. |
| 5+ | 5–10% | Long manifests / serial lists. |

### 6.2 Type co-occurrence matrix

Each pairwise co-occurrence must appear in at least 30 dev records:

|  | PERSON | ORG | ADDRESS | COMMODITY |
|---|---|---|---|---|
| **PERSON** | 30 | 50 (signer + co.) | 30 (c/o address) | 50 (signer + commodity) |
| **ORG** | — | 30 (multi-party deals) | 75 (buyer + ship-to) | 100 (every invoice) |
| **ADDRESS** | — | — | 30 (multi-address) | 50 (ship-to + commodity) |
| **COMMODITY** | — | — | — | 75 (serial commodity lists) |

### 6.3 Repeated entities

- Same surface form twice in one record: `Acme Trading | Acme Trading`.
- Same entity referenced by alias: `Acme Trading Co. (Acme)`.
- Two different commodities with same head noun: `red wine, white wine`.

### 6.4 Adjacency

- No separator between entities: `Maria GonzalezAcme Trading` (rare,
  but happens in OCR'd headers without spacing).
- Tight separator: `Maria Gonzalez|Acme Trading`.
- Comma-separated: `Maria Gonzalez, Acme Trading`.
- Punctuation only: `Maria Gonzalez. Acme Trading.`.

---

## 7. Noise and real-world artifact coverage

Every dev/test record is sampled from real input streams. Train must
include synthetic noise that matches each of these (via `ner/data/noise.py`):

### 7.1 OCR-style noise

- Random char-level typos (1–5% rate).
- Dropped punctuation (commas, periods, semicolons).
- Dropped spaces (`MariaGonzalez`, `AcmeTrading`).
- Confusable substitutions: `0` ↔ `O`, `1` ↔ `l` ↔ `I`, `5` ↔ `S`.
- Repeated characters: `Acmme Tradding`.
- Line-break artifacts mid-entity.

### 7.2 Fast-typing / messaging noise

- All-lowercase entire record.
- ALL-CAPS entire record.
- Missing terminal punctuation.
- Run-on sentences with no commas.
- Abbreviated entity types: `Co` (no period), `Ltd` (no period).

### 7.3 PDF / Word / Excel paste noise

- NBSP (` `) inside entity names.
- Zero-width chars (ZWSP `​`, ZWJ `‍`, BOM `﻿`).
- Soft hyphens (`­`) inside words.
- Smart quotes: `"` `"` `'` `'` instead of straight quotes.
- Em-dash / en-dash (`—` `–`) instead of hyphen.
- Ideographic space (`　`) between fields.

### 7.4 Tabular formatting

- Pipe-delimited: `Field1 | Field2 | Field3`.
- Tab-delimited: `Field1\tField2\tField3`.
- Key:value style: `Buyer: Acme Trading | Ship-to: 7 Canal St`.
- Header lines vs data lines.
- Misaligned columns (column drift due to long values).

### 7.5 Multi-language artifacts (English-primary domain)

- Currency symbols inline: `₹`, `€`, `¥`, `£`, `$`, `R$`, `CHF`.
- Mixed-script company names: `Mitsubishi 三菱`.
- Diacritics: `Köln`, `São Paulo`, `Türkiye`, `Köbenhavn`.
- Right-to-left fragments embedded in left-to-right context (Arabic,
  Hebrew company names).

### 7.6 Length extremes

- Empty input: `""`.
- Whitespace only: `"   "`.
- Single token: `"Acme"`.
- Maximum input (500 chars): full-length record at the truncation
  boundary, with an entity *straddling* position 500.
- Beyond max: 1,000-char record; verify nothing past 500 is scored.

---

## 8. Boundary discipline

These rules are mandatory at labeling time. Annotators reject records that
violate the rule rather than label them ambiguously.

### 8.1 COMMODITY

- **Exclude** quantity, unit, packaging:
  - `500 tons of robusta coffee` → span = `robusta coffee` (not `500 tons of robusta coffee`).
  - `12,000 kg HDPE resin` → span = `HDPE resin`.
  - `300 bales of cotton` → span = `cotton`.
- **Include** grade, treatment, variety, alphanumeric product codes that
  are inseparable from the commodity identity:
  - `Grade A robusta coffee` → all 4 words in span.
  - `304 stainless steel sheet` → all 4 tokens in span.
  - `Polyethylene resin HDPE` → all 3 tokens in span.
- **Exclude** intent / sentiment / verbs: "shipped robusta coffee" — `shipped`
  is outside the span.

### 8.2 ADDRESS

- **Include** the fullest contiguous locational string: street number, street
  name, city, state, postal code, country.
- **Exclude** leading prepositions and descriptors: "shipped to 42 Industrial
  Park Road" — span starts at `42`, not `to`.
- **Multi-line addresses** (`\n` separators) are a single contiguous span
  unless the address is genuinely two separate addresses.
- **Apartment / suite** numbers are part of the span if contiguous.

### 8.3 PERSON

- **Exclude** titles unless they are part of the legal name:
  - `Dr. Smith` → `Smith`.
  - `Mr. John Smith` → `John Smith`.
  - `John Smith MD` → `John Smith` (MD is a credential, not the name).
- **Include** generational suffixes when not comma-separated:
  - `John Smith Jr.` → `John Smith Jr.`.
  - `John Smith, Jr.` → `John Smith` (comma separates).
- **Exclude** job roles even when adjacent: `CEO Maria Gonzalez` → `Maria
  Gonzalez`.
- **Mononyms** (single-name people, e.g., `Madonna`, `Pelé`) are PERSON if
  context is unambiguous; flag for adjudication otherwise.

### 8.4 ORG

- **Include** legal-entity suffix when contiguous: `Co.`, `Inc.`, `Ltd`,
  `GmbH` are part of the span.
- **Exclude** the suffix only when it's clearly separable (parenthetical or
  after a comma).
- **Brand names** that double as PERSON / common nouns (`Tesla`, `Ford`,
  `Apple`) are ORG when the role-position is buyer / seller / vendor;
  PERSON when the role is signer / contact.
- **Acronyms** (`IBM`, `BASF`, `KPMG`) are ORG even when standalone.

---

## 9. Adversarial / ambiguity coverage

These cases are sampled deliberately because the model's accepted error
mode lives here. Every split must have at least 75 records in this
category, evenly split across rows:

| Row | Example | Expected label |
|---|---|---|
| Eponymous brand vs. founder | "Hermès handbag" | ORG |
| Eponymous brand vs. founder | "signed by Hermès Lopez" | PERSON |
| Person name as commodity modifier | "Maria Gonzalez coffee" | ORG (brand-on-commodity) |
| Geographic name as ORG | "Patagonia ordered ..." | ORG |
| Geographic name as part of ADDRESS | "shipped to Patagonia, Argentina" | ADDRESS |
| All-caps OCR'd ORG vs ALL-CAPS PERSON | `JOHN SMITH` vs `JOHN SMITH LTD` | suffix decides |
| Single-token entity | `Apple` (no context) | mark as ambiguous; require ≥ 1 surrounding signal |
| Two consecutive PERSONs | `Maria Gonzalez Felix Yu` (no separator) | two distinct PERSON spans |
| Number-prefix commodity | `42 robusta` | flag for context: number could be lot id or quantity |
| Frozen compound vs. real negation | `sugar-free vs. no sugar` | first POS, second NEG |

---

## 10. Zero-entity (true-negative) records

Records that contain no entities are essential — they teach the model when
NOT to fire. The dev set must include 100 such records covering:

- Pure descriptive prose: "The shipment arrived on time."
- Generic categories without proper-noun mention: "the coffee was bitter".
- Mentioned things that look entity-like but aren't:
  - Email signatures with no real entity: `Best regards,` `Sent from my iPhone`.
  - Boilerplate: `This is an automated message. Do not reply.`.
- Records with only timestamps / IDs: `2024-09-21 14:32 UTC — REF#88231`.

These records must produce ZERO predicted spans. The metric penalizes any
prediction on a zero-entity record as a false positive.

---

## 11. Diversity and bias coverage

Demographic / geographic / industry diversity is non-optional. Bias here
becomes a fairness liability in production.

### 11.1 PERSON name diversity (dev set)

- Western European names: ≥ 30.
- East Asian names (Chinese, Japanese, Korean): ≥ 30.
- South Asian names (Indian, Pakistani, Bangladeshi, Sri Lankan): ≥ 30.
- Middle Eastern / North African names: ≥ 20.
- African names (sub-Saharan): ≥ 20.
- Latin American names: ≥ 25.
- Pacific Islander names: ≥ 10.
- Approximate gender balance per region.

### 11.2 ADDRESS diversity

- US, UK, EU, India, China, Japan, Singapore, Australia, Brazil, Mexico,
  South Africa, Saudi Arabia, UAE — each at least 15 records.

### 11.3 COMMODITY industry diversity

Core verticals each ≥ 30 records:

- Energy (crude, LNG, refined products).
- Metals (steel, copper, aluminum, zinc, precious metals).
- Agricultural (coffee, sugar, cotton, grain, palm oil, cocoa).
- Chemical / plastics (HDPE, LDPE, ammonia, urea).
- Consumer goods (electronics, apparel, packaged foods).
- Pharmaceuticals (APIs, finished dosage).

### 11.4 ORG diversity

- Global multinationals: 30.
- Regional traders: 30.
- Small-business / SMEs: 30.
- Government / state-owned entities: 15.

---

## 12. Source mix

Dev and test should mirror the production input distribution. Track the
mix and target this breakdown (re-baseline annually):

| Source | % of dev | Distinguishing features |
|---|---|---|
| Invoices (header + line items) | 35% | Tabular, totals, dates, PO numbers |
| Manifests / packing lists | 25% | Serial commodity lists, addresses, line breaks |
| Emails (free prose) | 20% | Long sentences, signatures, multi-party threads |
| Webhook / API payloads (escaped text) | 10% | JSON-escaped strings, code-fenced |
| OCR'd PDFs | 10% | OCR errors, column drift, mis-segmented lines |

If production traffic moves significantly off this mix, the dev/test must
be re-balanced before the next release.

---

## 13. Quality process

### 13.1 Annotator setup

- Two independent annotators per record.
- Third adjudicator resolves disagreements; adjudicated labels are gold.
- Annotators see the labeling rubric (boundary rules from Section 8) at
  every session.

### 13.2 Inter-annotator agreement (IAA)

- Compute Cohen's κ over BIO labels per token (annotators agree on
  per-token label = agreement).
- Compute strict span F1 between annotators.
- Targets: κ ≥ 0.85, span F1 ≥ 0.90 before adjudication.
- If IAA drops below the floor, pause labeling and revisit the rubric.

### 13.3 Adjudication cadence

- Adjudicate every batch of 100 records before adding to dev/test.
- Adjudicator can reject a record (rubric violation, ambiguous context)
  — rejected records do not enter the dataset.

### 13.4 Span-offset verification

- Every record loaded by Python must satisfy `text[start:end] == ent.text`.
- A record failing this check is **rejected at load time**, not patched —
  the labeling tool is the source of truth.

---

## 14. Split strategy

### 14.1 Random vs. stratified

Stratified by (`source`, `dominant entity type`, `polarity_present`).
Random within each stratum.

### 14.2 Temporal

When records carry a timestamp:

- Dev: records from the trailing 6 months.
- Test: records from the most recent 1 month, held back until release.
- Train (synthetic): generated against pools current as of build time.

### 14.3 Cross-contamination guard

- Hash-based dedup between dev and test.
- Token-Jaccard ≥ 0.9 records flagged; manually placed on one side only.
- Records sharing the same upstream document (e.g., same PDF, multiple
  pages) all go to the same side.

---

## 15. Acceptance gates (before a release can ship)

The release cannot proceed if any gate fails:

1. **Dev size**: ≥ 500 records, all bucket minimums met (Sections 3, 5, 6).
2. **Test size**: ≥ 500 records, disjoint from dev, untouched since
   labeling.
3. **IAA**: κ ≥ 0.85 sampled across the latest 200 records.
4. **Offset invariant**: 0 records fail Python load-time verification.
5. **Boundary discipline audit**: 50 random dev records re-checked against
   Section 8 rules; ≥ 95% pass.
6. **Zero-entity coverage**: ≥ 100 true-negative records in dev with
   model false-positive rate < 5%.
7. **Polarity coverage**: ≥ 150 NEG_COMMODITY records, ≥ 50 frozen-compound
   positives.
8. **Diversity audit**: Section 11 minimums met.

---

## 16. Synthetic generation acceptance (train)

Before promoting a new train.jsonl:

1. Pools loaded from SQLite, all entity-type pools non-empty.
2. Templates parsed without `SlotFillError`.
3. `Record.validate()` passes on every emitted record (Python checks
   substring invariant).
4. NEG/POS commodity ratio in the synthetic set in `[0.15, 0.45]` —
   too low and the model under-learns negation; too high and it
   over-fires NEG.
5. At least one record per template after sampling.
6. Per-source style diversity present (tabular templates, prose
   templates, manifest templates).
7. Preprocessing applied identically (`preprocess.json` saved next to
   train.jsonl).

---

## 17. Edge-case smoke suite (frozen)

A separate, tiny dataset that lives forever and never changes — used as
a fast canary before every code change:

| ID | Text (verbatim) | Expected entities |
|---|---|---|
| EC-01 | `` (empty) | none |
| EC-02 | `   ` | none |
| EC-03 | `Acme` | one ORG-vs-PERSON ambiguous; document expected resolution |
| EC-04 | `Maria Gonzalez from Acme Trading Co. confirmed refined copper cathode shipped to 42 Industrial Park Road, Rotterdam, 3011 AB.` | the four entities from the TDD canonical example |
| EC-05 | `Manifest: galvanized steel coil, anhydrous ammonia, raw cane sugar — ETA Felix Yu, Oceanic Freight Co.` | three commodities + person + org |
| EC-06 | `does not contain wood` | one NEG COMMODITY: `wood` |
| EC-07 | `no special wood, but ordinary wood is fine` | NEG `special wood` + POS `wood` |
| EC-08 | `shipment of sugar-free chocolate and stainless steel sheet` | two POS commodities; cues dissolved |
| EC-09 | record with NBSP and zero-width chars inside `Maria Gonzalez​` | one PERSON; runtime returns original-coord offsets |
| EC-10 | 500-character record with last entity straddling position 500 | entity preserved, truncation aware |

These are checked-in as a unit test (`tests/test_edge_cases.py`), loaded
into the dev set as well, and never modified. Any change in their
predictions is a regression alert.

---

## 18. Maintenance

### 18.1 Refresh cadence

- **Train**: regenerate on every pool / template / preprocess change.
- **Dev**: add new records quarterly; never remove unless rubric was
  found to be wrong (and document the removal).
- **Test**: refresh annually with the latest 6-month window of real
  records; archive the old test set for longitudinal comparison.

### 18.2 Active-learning integration

- Production prediction logs (sampled at 0.1%) are surfaced for review
  weekly.
- High-loss / low-confidence predictions go to the labeling queue first.
- Newly-labeled records flow into dev; never directly into train (avoids
  the leakage path).

### 18.3 Drift detection

- Compute per-month token-distribution KL divergence vs. the dev set.
- If KL ≥ 0.05 sustained over 4 weeks, trigger a dataset refresh.

### 18.4 Versioning

Every dataset has:

- A semver tag (`dev-v2.3.0`).
- A content hash (SHA-256 of sorted JSONL).
- A datasheet recording: window covered, source mix, label-pass
  date, adjudicator IDs, IAA scores.
- An entry in the model registry so trained checkpoints trace back
  to the exact dataset version.

---

## 19. File layout and conventions

```
data/
  pools.sqlite                # entity / decoy / template pools (synthetic input)
  train.jsonl                 # synthetic; produced by scripts.generate_data
  preprocess.json             # frozen preprocess config used to produce train
  dev/
    records.jsonl             # hand-labeled dev
    datasheet.md              # window, sources, IAA, version
    annotators.txt            # who labeled, who adjudicated
  test/
    records.jsonl             # locked; opened once per release
    datasheet.md
    seal.txt                  # SHA-256 of records.jsonl + lock date
  edge_cases.jsonl            # the frozen smoke suite (Section 17)
```

JSONL row shape (Section 3 of the system code):

```json
{
  "text": "...",
  "entities": [
    {"type": "COMMODITY", "text": "...", "start": 0, "end": 0, "polarity": "POS"}
  ],
  "meta": {
    "source": "invoice|manifest|email|webhook|ocr",
    "split_stratum": "...",
    "annotator_ids": ["a1", "a2"],
    "adjudicator_id": "a3",
    "iaa_kappa": 0.91,
    "version": "dev-v2.3.0"
  }
}
```

`meta.preserve_spans` is reserved for synthetic records (negation-cue
ranges) and is not used in dev/test.

---

## 20. Open questions to resolve before first labeling pass

These are deliberately left ambiguous in the spec because they're product
calls, not technical ones. Lock answers in `docs/labeling_rubric.md`
before starting:

1. Are job titles preceding a name ever part of the PERSON span?
   (Default: no.)
2. Is the country always included in ADDRESS, even when the rest of the
   address is domestic and the country is implicit? (Default: yes if the
   country is written; do not add it if absent.)
3. Are partner / co-signer pairs (`Maria Gonzalez & John Smith`)
   one PERSON span or two? (Default: two.)
4. Are pluralized brand mentions (`Toyotas`) ORG? (Default: yes; the `s`
   is part of the span.)
5. Are stock / product codes (`HDPE 1810D`, `LDPE 5500`) part of the
   COMMODITY span when separated by a space? (Default: yes if no other
   token sits between.)
6. For "may contain X" (uncertain assertion) — out of scope for v1;
   labeled as POS but flagged in `meta.uncertain=true` for future
   `UNCERTAIN` label expansion.
