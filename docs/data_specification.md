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

### 4.5 PERSON names — regional patterns and conventions

Every dev/test split must include records from each region. Order column
reflects the canonical written order in the home script; production text
frequently appears in Western order regardless of origin — both forms are
labeled. Diacritics, particles, and patronymics are part of the span.

| Region / culture | Order | Example | Span-affecting notes |
|---|---|---|---|
| Anglo (US/UK/AU/NZ/CA) | Given + (middle) + family | `John Robert Smith` | Double-barreled `Smith-Jones`; titles excluded |
| Anglo (Irish/Scottish) | Particle + family | `O'Brien`, `MacDonald`, `Mac an Bhaird` | Apostrophe variants; `Mc` vs `Mac` casing |
| French | Given + family with particles | `Jean-Paul de la Croix` | `de`, `du`, `de la`, `le` particles in span |
| Spanish | Given + paternal + maternal | `Juan García López` | Two surnames standard; `de`, `del` particles |
| Portuguese (PT/BR) | Given + maternal + paternal | `João Silva Santos`, `Luiz Inácio Lula da Silva` | 3+ given names common; `da`, `dos`, `de` |
| Italian | Given + family with particle | `Marco di Stefano` | `di`, `da`, `della`, `del` particles |
| German / Austrian / Swiss-German | Given + family | `Hans Müller`, `Maria von Habsburg` | Umlauts `ä ö ü ß`; `von`, `zu`, `von der` |
| Dutch / Flemish | Given + particle + family | `Pieter van der Berg` | Multi-word particles `van der`, `van den`, `van het` |
| Swedish / Norwegian / Danish | Given + family | `Lars Eriksson`, `Astrid Hansen` | `-sson`, `-sen`, `-strup` endings |
| Icelandic | Given + patronymic / matronymic | `Björk Guðmundsdóttir`, `Magnús Ólafsson` | `-son` / `-dóttir` derived from parent's given |
| Finnish | Given + family | `Mika Häkkinen`, `Sanna Marin` | Umlauts; double consonants |
| Polish | Given + family | `Wojciech Szczęsny`, `Anna Kowalska` | `-ski`/`-cki` (m), `-ska`/`-cka` (f); diacritics `ł ą ę ć ń ó ś ź ż` |
| Czech / Slovak | Given + family | `Karel Novák`, `Marie Nováková` | Female form `-ová` suffix |
| Hungarian | **Family + given** in source | `Bartók Béla` (HU) / `Béla Bartók` (West) | Document expected order at labeling time |
| Russian | Given + patronymic + family | `Vladimir Ilyich Lenin` | `-ovich`/`-ovna` patronymic; `-ov`/`-ova` family |
| Ukrainian | Given + family | `Volodymyr Zelenskyy` | `-enko`, `-uk`, `-yi`; transliteration varies |
| Greek | Given + family | `Konstantinos Papadopoulos` | `-opoulos`, `-idis`, `-akis` |
| Romanian | Given + family | `Ion Popescu` | `-escu`, `-eanu` |
| Bulgarian / Serbian / Croatian | Given + family | `Goran Bregović`, `Ivan Petrov` | `-ić`, `-ov`, `-ev` |
| Turkish | Given + family | `Mustafa Kemal Atatürk` | `ç ş ğ ı ö ü`; family names adopted 1934 |
| Hebrew / Israeli | Given + family | `David ben Gurion`, `Tzipi Livni` | `ben` (son of), `bat` (daughter of) particles |
| Arabic — Levantine / Gulf | Given + father + grandfather + tribal | `Mohammed bin Salman Al Saud` | `bin`/`ibn`, `bint`, `Al` (tribe), full chain in span |
| Arabic — kunya form | Honorific + child-of-X | `Abu Mazen`, `Umm Khalid` | `Abu` (father of), `Umm` (mother of) |
| Persian / Farsi | Given + (middle) + family | `Mohammad Ali Khamenei` | Family names ~1925; multi-part |
| Chinese (Mandarin) | **Family + given** | `Wang Wei`, `Li Na` | Family 1–2 chars; Pinyin / Wade-Giles / Yale romanizations |
| Chinese (Cantonese / HK) | Family + given (different romanization) | `Wong Ka-Wai`, `Chan Ho-Man` | Hyphenated given; Cantonese phonetics |
| Japanese | **Family + given** (JP); reversed in West | `Yamamoto Tarō` / `Tarō Yamamoto` | Long marks `ō ū`; sometimes dropped to ASCII |
| Korean | **Family + given** | `Kim Min-jun`, `Park Ji-sung` | Single-syllable family; hyphenated given; `Kim`/`Gim`, `Lee`/`Yi` variants |
| Vietnamese | Family + middle + given | `Nguyễn Văn Anh` | `Nguyễn`, `Trần`, `Lê` top families; diacritics critical |
| Thai | Given + family + (nickname) | `Somchai Sirikhom (Chai)` | Family names adopted 1913; very long names; nicknames in parens |
| Indonesian / Malay | Mononym or given-only | `Sukarno`, `Joko Widodo` | One-word names common; `bin`/`binti` optional |
| Filipino | Spanish-influenced; `y` between surnames | `Juan Dela Cruz`, `María Santos y Reyes` | `Dela`, `Del`, `De Los`, `Y` particles |
| Indian — North | Given + family | `Aarav Kumar`, `Priya Sharma` | Families: `Kumar`, `Sharma`, `Singh`, `Gupta`, `Verma`, `Mishra`, `Tiwari` |
| Indian — South Tamil | Initial(s) + given (no family) | `A.R. Rahman`, `M. Karunanidhi` | Initials = father's name / village; period spacing varies |
| Indian — South Telugu / Kannada / Malayalam | Initial(s) + given | `K. Chandrasekhara Rao`, `M. K. Stalin` | Initials before given; multiple initials common |
| Indian — Sikh | Given + Singh / Kaur (+ optional family) | `Harpreet Singh`, `Simran Kaur Sandhu` | `Singh` (m), `Kaur` (f) standard middle/last |
| Indian — Bengali | Given + family | `Soumya Chakraborty`, `Rabindranath Tagore` | Long families: `Chakraborty`, `Banerjee`, `Mukherjee`, `Bhattacharya` |
| Indian — Maharashtrian | Given + father's given + family | `Sachin Ramesh Tendulkar` | Three-part name common |
| Indian — Gujarati / Parsi | Given + family | `Narendra Modi`, `Ratan Tata` | `Patel`, `Shah`, `Mehta`, `Tata` |
| Pakistani | Given + (middle) + family / tribal | `Muhammad Ali Khan` | `Khan`, `Sheikh`, `Sayyid`, `Mirza`, `Qureshi` |
| Sri Lankan Sinhala | Initial(s) + given + family | `D.M. Jayaratne` | Long compound families: `Wickremasinghe`, `Rajapaksa` |
| Sri Lankan Tamil | Initial + given | `M. Karunanidhi` | Same pattern as Indian Tamil |
| Bangladeshi | Given + family | `Sheikh Hasina`, `Md. Karim` | `Md.` for `Mohammad` very common (abbreviation in span) |
| Nepalese / Bhutanese | Given + family | `Pushpa Kamal Dahal`, `Karma Lhamo` | Tibetan Buddhist & Hindu name traditions |
| Yoruba (Nigeria) | Given + family | `Adeyemi Adebayo`, `Wole Soyinka` | `-yemi`, `-bayo`, `Ade-`, `Olu-` prefixes |
| Igbo (Nigeria) | Given + family | `Chukwuemeka Ojukwu`, `Chinua Achebe` | `Chukwu-`, `Nna-`, `Ada-` prefixes |
| Hausa | Given + father's given | `Mohammed Buhari`, `Aliyu Wamakko` | Islamic-influenced |
| Ethiopian / Eritrean | Given + father's given (no family) | `Abebe Bikila`, `Haile Selassie`, `Tewodros Adhanom Ghebreyesus` | "Last name" = father's given; no Western family-name concept |
| Somali | Triple patronymic | `Mohammed Abdullahi Mohammed` | Three given names: own + father + grandfather |
| Swahili (East Africa) | Given + family | `Uhuru Kenyatta`, `Julius Nyerere` | Bantu + Arabic mix |
| Afrikaans (South Africa) | Given + particle + family | `Jan van der Merwe`, `Pieter du Toit` | Dutch-derived particles |
| Zulu / Xhosa / Sotho | Given + family | `Nelson Rolihlahla Mandela`, `Cyril Ramaphosa` | Click consonants in some names (`!`, `\|`, `c`, `q`, `x`) |
| West African Francophone | Given + family | `Aminata Diallo`, `Aliou Touré`, `Ousmane Sembène` | French diacritics; families `Diallo`, `Touré`, `Diop`, `Cissé` |
| Mexican | Given + paternal + maternal | `Carlos García Rodríguez`, `Frida Kahlo y Calderón` | Two surnames; `de` joiner; `Sr.`/`Jr.` |
| Argentinian / Uruguayan | Italian-influenced; given + family | `Diego Armando Maradona`, `Lionel Andrés Messi` | Italian + Spanish mix |
| Brazilian | Multiple given + maternal + paternal | `Luiz Inácio Lula da Silva` | Often 3+ given names; `da`, `dos`, `de` particles |
| Cuban / Caribbean | Spanish + African mix | `Pedro Pérez Cabrera`, `Celia Cruz` | Compound surnames |
| Indigenous Mesoamerican | Mayan / Nahuatl / Quechua names | `Cuauhtémoc`, `Atahualpa` | Often single name; modernized forms add Spanish surname |
| Hawaiian | Given + family | `Kamehameha`, `Liliʻuokalani`, `Bernice Pauahi Bishop` | `ʻ` (okina) and macrons `ā ē ī ō ū` |
| Maori (NZ) | Given + family | `Tāmaki Mākaurau`, `Witi Ihimaera` | Macrons; often single given name |
| Polynesian (Samoan, Tongan, Fijian) | Given + family | `Sione Filitonga`, `Tupou Vaipulu` | Long, vowel-heavy |
| Mixed-heritage / diasporic | Hyphenated or stacked | `Mary Chen-Smith`, `Lily Wong (Wang Lihua)` | Romanization + native parens common in dev set |

### 4.6 PERSON honorifics, titles, and credentials

Pre-name honorifics — **excluded from span** unless inextricable from the
legal name:

- **Western**: `Mr.`, `Mrs.`, `Ms.`, `Miss`, `Dr.`, `Prof.`, `Hon.`, `Rev.`, `Sir`, `Dame`, `Lord`, `Lady`, `Mx.`
- **Religious**: `Father`/`Fr.`, `Sister`/`Sr.`, `Brother`/`Br.`, `Rabbi`, `Cantor`, `Imam`, `Sheikh`, `Bhante`, `Lama`, `Swami`, `Pope`
- **Military / law-enforcement**: `Gen.`, `Maj.`, `Col.`, `Lt.`, `Capt.`, `Sgt.`, `Pvt.`, `Cpl.`, `Adm.`, `Cmdr.`, `Det.`, `Off.`
- **Indian / South Asian**: `Shri`, `Sri`, `Smt.`, `Kumari`, `Pandit`, `Acharya`, `Swami`, `Babu`, `Sahib`
- **Arabic / Islamic**: `Sheikh`, `Hajji`, `Sayyid`, `Mawlana`, `Hafiz`, `Mufti`
- **Spanish / Latin**: `Don`, `Doña`, `Sr.`, `Sra.`, `Srta.`, `Lic.`, `Ing.`, `Arq.`
- **German**: `Herr`, `Frau`, `Doktor`, `Professor`, `Doktor-Ingenieur` / `Dr.-Ing.`
- **Japanese / Chinese / Korean address suffixes** (`-san`, `-sama`, `-sensei`, `-kun`, `-chan`, `-ssi`, `-nim`, `-laoshi`): excluded; almost always written separately
- **Address-form prefixes**: `Dear`, `To`, `Attn:`, `c/o`

Post-name elements:

- **Generational** (included when no separating comma): `Jr.`, `Sr.`, `II`, `III`, `IV`, `V`. Example: `John Smith Jr.` → span includes `Jr.`; `John Smith, Jr.` → span ends at `Smith`.
- **Academic / professional credentials** (excluded): `PhD`, `Ph.D.`, `MD`, `M.D.`, `DDS`, `DVM`, `JD`, `MBA`, `MS`, `BS`, `BA`, `RN`, `PE`, `CPA`, `CFA`, `CFP`, `Esq.`, `Esquire`, `LLM`, `LL.B.`
- **Religious orders** (excluded): `OFM`, `SJ`, `OSB`
- **Civil service** (excluded): `IAS`, `IPS`, `IFS`, `IRS` (Indian); `IES`
- **Honours** (excluded): `OBE`, `MBE`, `CBE`, `KBE`, `KCMG`, `KCB`, `DSO`, `VC` (UK); `Bharat Ratna`, `Padma Bhushan` (India)
- **Military post-name**: `(Ret.)`, `(USAF)`, `(USMC)` — excluded

### 4.7 ORG legal-entity suffixes by jurisdiction

Every dev/test split must include at least 5 records for each row of this
table; the synthetic train pool must support all of them as decoys or
legitimate suffixes. Periods, casing, and hyphens vary in production text
— accept the variant forms; the canonical form is the row label.

| Region | Suffixes | Notes |
|---|---|---|
| **USA** | `Inc.`, `Corp.`, `Co.`, `LLC`, `L.L.C.`, `LLP`, `LP`, `PC`, `P.C.`, `PLLC`, `PA`, `Holdings` | Periods optional in modern usage |
| **UK** | `Ltd`, `Limited`, `PLC`, `plc`, `LLP`, `& Co.`, `Holdings` | `plc` is listed; `Ltd` is private |
| **Ireland** | `Ltd`, `DAC`, `CLG`, `PLC`, `Teoranta` (Gaelic) | |
| **Germany** | `GmbH`, `AG`, `KG`, `KGaA`, `OHG`, `SE`, `Stiftung`, `gGmbH`, `UG (haftungsbeschränkt)` | `UG` for low-capital |
| **Austria** | `GmbH`, `AG`, `KG`, `OG`, `GesmbH` | |
| **Switzerland** | `AG`, `SA`, `Sàrl`, `GmbH`, `Genossenschaft` | German / French / Italian variants |
| **France** | `SA`, `SAS`, `SARL`, `SCA`, `SCS`, `EURL`, `SNC`, `SCEA`, `SCP` | `SAS`, `SARL` most modern |
| **Belgium / Luxembourg** | `SA`, `SPRL`, `BVBA`, `CVBA`, `SCRL`, `NV`, `BV` | Bilingual |
| **Netherlands** | `BV`, `NV`, `CV`, `Coöperatie`, `Stichting` | |
| **Spain** | `S.A.`, `SA`, `S.L.`, `SL`, `SLU`, `Sdad. Coop.` | |
| **Portugal** | `SA`, `Lda.`, `Sociedade Unipessoal` | `Lda.` ubiquitous |
| **Italy** | `S.p.A.`, `S.r.l.`, `S.a.s.`, `S.n.c.`, `S.c.a.r.l.` | Periods usually present |
| **Sweden** | `AB`, `HB`, `KB`, `ekonomisk förening` | `AB` covers most |
| **Norway** | `AS`, `ASA`, `ANS`, `DA`, `BA` | |
| **Denmark** | `A/S`, `ApS`, `K/S`, `I/S`, `P/S` | Forward slash in suffix |
| **Finland** | `Oy`, `Oyj`, `Ky`, `Tmi`, `Osk` | `Oyj` listed; `Oy` private |
| **Iceland** | `ehf.`, `hf.`, `slhf.`, `sf.` | |
| **Poland** | `Sp. z o.o.`, `S.A.`, `Sp. j.`, `S.K.A.` | Long with internal spaces |
| **Czech / Slovakia** | `s.r.o.`, `a.s.`, `k.s.`, `v.o.s.` | |
| **Hungary** | `Kft.`, `Zrt.`, `Nyrt.`, `Bt.`, `Kkt.` | |
| **Romania** | `SRL`, `SA`, `SCS`, `SCA`, `PFA` | |
| **Greece** | `AE`, `EPE`, `OE`, `EE`, `IKE` | |
| **Russia** | `OOO`, `OAO`, `PAO`, `ZAO`, `IP`, `AO` | Cyrillic + Latin variants (`ООО` ↔ `OOO`) |
| **Ukraine** | `TOV`, `PrAT`, `PAT`, `FOP` | |
| **Turkey** | `A.Ş.`, `Ltd. Şti.`, `S.M.M.M.` | `Şti.` = `Şirket`; non-ASCII `Ş` |
| **Israel** | `Ltd`, `Bet"M`, `R.A.`, `Aguda` | Hebrew abbreviation `Bet"M` common |
| **Saudi Arabia** | `Co. Ltd.`, `LLC`, `JSC` | Aramco state-owned |
| **UAE** | `LLC`, `FZE`, `FZ-LLC`, `DMCC`, `JLT`, `PJSC` | Free-zone variants critical to coverage |
| **Egypt** | `S.A.E.`, `LLC`, `Joint Stock Co.` | |
| **Morocco / Algeria / Tunisia** | `SA`, `SARL`, `SCA` | French-derived |
| **Nigeria** | `Plc`, `Ltd`, `Co. Ltd.` | |
| **Kenya** | `Ltd`, `Limited`, `Plc`, `Co.` | |
| **South Africa** | `Pty Ltd`, `(Pty) Ltd`, `Proprietary Limited`, `Ltd`, `Inc.`, `CC` | `CC` = close corporation |
| **Ghana** | `Ltd`, `Plc`, `Co. Ltd.` | |
| **China (mainland)** | `Co., Ltd.`, `公司`, `有限公司`, `集团` (Group), `控股` (Holdings) | English + Chinese name often both present |
| **Hong Kong** | `Ltd`, `Limited`, `Co. Ltd.`, `(HK)` | Bilingual EN/ZH |
| **Taiwan** | `Co., Ltd.`, `Inc.`, `股份有限公司` | |
| **Japan** | `K.K.` (株式会社), `Co., Ltd.`, `Inc.`, `Y.K.` (有限会社), `G.K.` (合同会社) | `G.K.` is LLC-equivalent |
| **South Korea** | `Co., Ltd.`, `Inc.`, `Corp.`, `주식회사` | Chaebol naming pattern |
| **Singapore** | `Pte Ltd`, `Pte. Ltd.`, `Limited`, `Pte`, `LLP` | `Pte` private, `Limited` public |
| **Malaysia** | `Sdn. Bhd.`, `Sdn Bhd`, `Berhad`, `Bhd.`, `PLT` | `Sdn Bhd` private; `Berhad` public |
| **Indonesia** | `PT`, `PT.`, `Tbk`, `Persero`, `CV` | `PT` prefix (uncommon position); `Tbk` listed |
| **Philippines** | `Inc.`, `Corp.`, `Corporation`, `Ltd.` | |
| **Thailand** | `Co., Ltd.`, `PCL`, `บริษัท ... จำกัด` | Thai script in legal name |
| **Vietnam** | `Ltd.`, `JSC`, `Co., Ltd.` | |
| **India** | `Pvt. Ltd.`, `Pvt Ltd`, `Private Limited`, `Ltd`, `Limited`, `LLP` | `Pvt. Ltd.` ubiquitous |
| **Pakistan** | `Pvt. Ltd.`, `(Pvt) Limited`, `Ltd`, `(SMC-Pvt) Ltd.` | |
| **Sri Lanka** | `PLC`, `Ltd`, `(Pvt) Ltd` | |
| **Bangladesh** | `Ltd.`, `Limited`, `(Pvt.) Ltd.`, `Group` | |
| **Brazil** | `Ltda.`, `S.A.`, `S/A`, `EIRELI`, `ME`, `EPP` | `S/A` forward-slash variant common |
| **Mexico** | `S.A.`, `S.A. de C.V.`, `S. de R.L. de C.V.`, `S.A.B. de C.V.` | `S.A.B.` listed |
| **Argentina** | `S.A.`, `SRL`, `SAS`, `Coop. Ltda.` | |
| **Chile** | `S.A.`, `Ltda.`, `SpA`, `EIRL` | |
| **Colombia** | `S.A.`, `S.A.S.`, `Ltda.` | `S.A.S.` is the modern default |
| **Peru** | `S.A.`, `S.A.C.`, `S.R.L.`, `E.I.R.L.` | |
| **Venezuela** | `C.A.`, `S.A.`, `S.R.L.` | `C.A.` = Compañía Anónima |
| **Canada** | `Inc.`, `Ltd.`, `Corp.`, `ULC`, `LLP` | English + French (e.g. `Ltée`) |
| **Australia** | `Pty Ltd`, `Pty. Ltd.`, `Limited`, `Ltd`, `Pty.` | |
| **New Zealand** | `Ltd`, `Limited` | |

### 4.8 BRAND naming patterns

Brand class is folded into ORG in this system (Section 1 of the design
record). The cases below must still be coverable as ORG spans; the
modifier-on-commodity pattern needs special attention.

**Single-token brands by casing convention:**

- All-lowercase by design: `eBay`, `iPhone`, `iPad`, `nike` (marketing), `mRNA`
- All-uppercase by design: `IBM`, `IKEA`, `BMW`, `BBVA`, `LG`, `HSBC`, `KFC`, `UNIQLO`, `ASML`, `BASF`
- CamelCase: `PayPal`, `WeChat`, `FedEx`, `YouTube`, `GitHub`, `LinkedIn`, `OpenAI`, `EasyJet`, `JetBlue`, `MasterCard`, `MetLife`
- Initial-lowercase CamelCase: `eBay`, `iPhone`, `iPod`, `mRNA`
- Vowel-light marketing: `Lyft`, `Tumblr`, `Flickr`, `Grindr`, `Fivvr`, `Rappi`

**Multi-word brands:**

- Hyphenated: `Coca-Cola`, `Bristol-Myers Squibb`, `Häagen-Dazs`, `Mercedes-Benz`, `Hewlett-Packard`, `Wal-Mart`
- Ampersand: `Procter & Gamble`, `Johnson & Johnson`, `Marks & Spencer`, `Dolce & Gabbana`, `H&M`, `AT&T`, `A&W`, `Ben & Jerry's`
- Slash: `S/MIME`, `OS/2`, `B/E Aerospace`
- Apostrophe: `McDonald's`, `Wendy's`, `Macy's`, `Sotheby's`, `Levi's`, `Hershey's`, `Reese's`
- Quoted: `Toys "R" Us`, `Books "A" Million`
- Exclamation: `Yahoo!`, `Reach!`
- Special chars: `E*TRADE`, `Joe's*Coffee`, `Asahi+`
- "The" prefix in legal name: `The North Face`, `The Coca-Cola Company`, `The Home Depot`, `The Boeing Company`, `The Walt Disney Company`
- Possessive: `Levi's`, `Macy's`, `Wendy's`, `Hershey's`, `Domino's`

**Numbered brands:**

- Leading number: `3M`, `7-Eleven`, `7-Up`, `21st Century Fox`, `99 Cents Only`, `5 Guys`
- Trailing number: `Forever 21`, `Heinz 57`, `Channel 4`
- Embedded number: `B&Q`, `J2`, `A1`

**Brands with foreign diacritics / non-ASCII:**

- French: `L'Oréal`, `Nestlé`, `Citroën`, `Renault`, `Carrefour`, `Crédit Agricole`
- Spanish: `Telefónica`, `BBVA`, `Inditex`, `Volkswagen España`
- German: `Müller`, `Lufthansa`, `Henkel`, `Bayer`, `Beiersdorf`
- Scandinavian: `Häagen-Dazs` (faux-Danish), `LEGO`, `Volvo`, `Ericsson`, `Spotify`, `IKEA`, `Ørsted`
- Japanese romanized: `Asahi`, `Kirin`, `Sapporo`, `Toyota`, `Honda`, `Mitsubishi`, `Sumitomo`, `Sumitomo Mitsui`
- Mandarin pinyin: `Huawei`, `Xiaomi`, `Lenovo`, `BYD`, `Geely`, `Haier`, `Alibaba`, `Tencent`
- Korean: `Samsung`, `LG`, `Hyundai`, `Kia`, `SK Hynix`, `POSCO`
- Russian / Cyrillic both forms: `Gazprom` / `Газпром`, `Lukoil` / `Лукойл`, `Sberbank` / `Сбербанк`, `Yandex` / `Яндекс`
- Arabic transliteration: `Aramco`, `Emirates`, `Etihad`, `Al-Futtaim`, `Al-Rajhi`, `STC`, `Saudi Telecom`
- Indian: `Tata`, `Reliance`, `Infosys`, `Wipro`, `Mahindra`, `Bharti Airtel`, `Adani`, `JSW`

**Brand portfolios — parent vs sub-brand:**

| Parent ORG | Sub-brands typically extracted as ORG (treated equivalently) |
|---|---|
| Apple Inc. | `Apple`, `iPhone`, `iPad`, `Mac`, `Apple Watch`, `MacBook` |
| Alphabet / Google | `Google`, `YouTube`, `Waymo`, `Verily`, `DeepMind`, `Pixel` |
| Meta Platforms | `Facebook`, `Instagram`, `WhatsApp`, `Messenger`, `Threads` |
| Procter & Gamble | `Tide`, `Pampers`, `Gillette`, `Olay`, `Crest`, `Bounty`, `Pantene`, `Vicks` |
| Unilever | `Dove`, `Knorr`, `Lipton`, `Hellmann's`, `Magnum`, `Axe`, `Surf`, `Lux` |
| PepsiCo | `Pepsi`, `Lay's`, `Doritos`, `Quaker`, `Gatorade`, `Tropicana`, `Mountain Dew` |
| Nestlé S.A. | `Nescafé`, `KitKat`, `Maggi`, `Häagen-Dazs`, `Perrier`, `Purina`, `Gerber` |
| Tata Group | `Tata Steel`, `TCS`, `Tata Motors`, `Jaguar Land Rover`, `Tata Consumer Products` |
| Samsung Group | `Samsung Electronics`, `Samsung C&T`, `Samsung Heavy Industries`, `Samsung SDS` |
| LVMH | `Louis Vuitton`, `Dior`, `Givenchy`, `Tiffany & Co.`, `Sephora`, `Bulgari`, `Fendi` |

**Brand-as-modifier on a commodity** (the disambiguation case from the design conversation):

- `Apple iPhone shipment` → ORG `Apple`, COMMODITY `iPhone` *(or* COMMODITY `Apple iPhone` *if product spans are merged; the rubric must pick one and stay consistent)*
- `Toyota Camry, 10 units` → ORG `Toyota`, COMMODITY `Camry`
- `Coca-Cola syrup, 500 kg` → ORG `Coca-Cola`, COMMODITY `syrup`
- `Häagen-Dazs ice cream` → ORG `Häagen-Dazs`, COMMODITY `ice cream`
- `Nestlé chocolate` → ORG `Nestlé`, COMMODITY `chocolate`
- `Patagonia jackets` → ORG `Patagonia`, COMMODITY `jackets`

### 4.9 ORG / BRAND / PERSON fuzzy and messy variants

Every variant pattern below must appear in dev/test. None of these are
"optional cleanup" cases — all of them are present in real production
inputs and the model has to handle them.

**Casing variants** (same legal name, different surface form):

| Canonical | Variant | Source |
|---|---|---|
| `Acme Trading Co.` | `ACME TRADING CO.` | OCR'd letterhead |
| `Acme Trading Co.` | `acme trading co` | Fast-typed email |
| `Acme Trading Co.` | `Acme Trading CO` | Mixed casing |
| `Acme Trading Co.` | `Acme trading co.` | Sentence-case mid-sentence |
| `IBM` | `Ibm`, `ibm`, `IBM.` | Capitalized as if a regular word |
| `iPhone` | `IPhone`, `iphone`, `IPHONE` | Wrong CamelCase variants |
| `eBay` | `Ebay`, `EBAY`, `ebay` | All variants present in invoices |
| `Maria Gonzalez` | `MARIA GONZALEZ`, `maria gonzalez`, `Maria GONZALEZ` | OCR / shouting / mixed |

**Punctuation drift in suffixes:**

| Canonical | Variants |
|---|---|
| `Inc.` | `Inc`, `INC`, `Incorporated`, `Inc,`, `Inc..` |
| `Co.` | `Co`, `Company`, `Comp.`, `C/o` |
| `Ltd.` | `Ltd`, `LTD`, `Limited`, `Ltd,` |
| `& Co.` | `and Co.`, `and Company`, `& Company`, `&Co.` |
| `S.A.` | `SA`, `S.A`, `S. A.`, `S/A` |
| `Pvt. Ltd.` | `Pvt Ltd`, `PVT LTD`, `(P) Ltd.`, `Pvt.Ltd.`, `P. Ltd.` |
| `Sdn. Bhd.` | `Sdn Bhd`, `Sdn.Bhd.`, `Sdn-Bhd`, `SDN BHD` |
| `Pte Ltd` | `Pte. Ltd.`, `Pte Ltd.`, `PTE LTD`, `Pte.Ltd` |
| `GmbH` | `gmbh`, `GMBH`, `G.m.b.H.` |
| `S.A. de C.V.` | `SA de CV`, `S.A. de CV`, `S.A.deC.V.` |

**Whitespace and separator drift:**

| Canonical | Variants |
|---|---|
| `Bristol-Myers Squibb` | `Bristol Myers Squibb`, `Bristol—Myers Squibb` (em-dash), `BristolMyers Squibb`, `Bristol  Myers  Squibb` |
| `Mercedes-Benz` | `Mercedes Benz`, `Mercedes-benz`, `MercedesBenz` |
| `Procter & Gamble` | `Procter and Gamble`, `Procter&Gamble`, `Procter &Gamble`, `Procter & gamble`, `Procter and gamble` |
| `Toys "R" Us` | `Toys R Us`, `Toys 'R' Us`, `Toys “R” Us` (smart quotes), `ToysRUs` |
| `Coca-Cola` | `Coca Cola`, `coca-cola`, `CocaCola`, `COCA-COLA`, `Coca‑Cola` (non-breaking hyphen) |
| `H&M` | `H & M`, `H and M`, `h&m`, `H AND M` |
| `Häagen-Dazs` | `Haagen-Dazs` (diacritic dropped), `Haagen Dazs`, `HAAGEN-DAZS`, `Häagen Dazs` |
| `L'Oréal` | `LOreal`, `L'Oreal` (accent dropped), `L Oreal`, `L'Oreal`, `L`Oreal` (smart quote) |
| `Maria Gonzalez` | `MariaGonzalez`, `Maria  Gonzalez` (double space), `Maria Gonzalez` (NBSP between) |

**Apostrophe variants** (straight, curly, missing, replaced):

| Canonical | Variants |
|---|---|
| `McDonald's` | `McDonalds`, `Mcdonalds`, `McDonald's` (curly `’`), `MCDONALD'S`, `MCDONALDS`, `McDonalds'` |
| `O'Brien & Co.` | `OBrien & Co.`, `O'Brien & Co.` (curly), `O Brien & Co.`, `O`Brien & Co.` |
| `Sotheby's` | `Sothebys`, `SOTHEBYS`, `Sotheby's` (curly) |

**Abbreviation / acronym variants:**

| Long form | Variants |
|---|---|
| `International Business Machines Corporation` | `IBM`, `Int'l Business Machines`, `Internat'l Business Machines Corp.`, `I.B.M.` |
| `J.P. Morgan Chase & Co.` | `JPMorgan Chase`, `JP Morgan`, `J.P.Morgan`, `JPMorganChase`, `JPM` |
| `Bank of America Corporation` | `BofA`, `Bank of America`, `BoA`, `B of A`, `BAC` |
| `Hewlett-Packard Company` | `HP`, `Hewlett Packard`, `H-P`, `HPE` (post-split entity), `Hewlett-Packard Co.` |
| `Standard Chartered Bank` | `StanChart`, `Standard Chartered`, `SCB`, `Standard Chartered plc` |
| `Industrial and Commercial Bank of China` | `ICBC`, `工商银行`, `Industrial & Commercial Bank` |
| `Tata Consultancy Services` | `TCS`, `Tata Consultancy`, `Tata Consultancy Services Ltd` |

**Brand vs legal name** (production text mixes both freely; both must be labeled):

| Brand (informal) | Legal name (formal) |
|---|---|
| `Google` | `Alphabet Inc.` / `Google LLC` |
| `Facebook` | `Meta Platforms, Inc.` |
| `Pringles` | `Kellanova` (formerly `Kellogg Company`) |
| `Snickers` | `Mars, Incorporated` |
| `Doritos` | `Frito-Lay, Inc.` / `PepsiCo, Inc.` |
| `iPhone` | `Apple Inc.` |
| `KitKat` (rest of world) | `Nestlé S.A.` |
| `KitKat` (US only) | `The Hershey Company` |
| `Oreo` | `Mondelēz International` |

**OCR-specific corruption (ORG / BRAND):**

| Original | OCR'd | Fault |
|---|---|---|
| `Acme` | `Acrne` | `m` → `rn` ligature confusion |
| `Co.` | `Cu.`, `Co,` | `o` → `u`, period → comma |
| `Ltd` | `Lid`, `Ltd` (l→1), `Ld`, `LtcI` | char confusion |
| `IBM` | `1BM`, `IBN`, `lBM`, `IRM` | `I` ↔ `1` ↔ `l`; `M` ↔ `N` ↔ `R` |
| `Patagonia` | `Pat agonia`, `Pat-agonia` | spurious space / hyphen |
| `Microsoft` | `MicroSoft`, `Micro soft`, `Microscft` | case / split / `o`→`c` |
| `Sumitomo` | `Sumltomo`, `Sumttomo` | `i` confusion |
| `Häagen-Dazs` | `Haaqen-Dazs` | `g` → `q` |
| `Aramco` | `Aramoo`, `Aramcc` | `c`/`o` confusion |

**Romanization / transliteration pairs** (both labeled when both appear in source):

| Canonical (English) | Variants |
|---|---|
| `Sinopec` | `China Petroleum & Chemical Corporation`, `中国石化` |
| `Gazprom` | `Газпром`, `OAO Gazprom`, `PJSC Gazprom`, `Public Joint Stock Company Gazprom` |
| `Lukoil` | `Лукойл`, `LUKOIL`, `OOO Lukoil`, `PAO Lukoil` |
| `Huawei` | `华为`, `HUAWEI`, `Huawei Technologies Co., Ltd.` |
| `Samsung` | `삼성`, `Samsung Electronics Co., Ltd.`, `SEC` |
| `Toyota` | `トヨタ`, `Toyota Motor Corporation`, `TMC`, `Toyota Jidosha` |
| `Hitachi` | `日立`, `Hitachi, Ltd.`, `(株) 日立製作所` |
| `Tata` | `Tata Group`, `Tata Sons Pvt. Ltd.`, `टाटा` |
| `Mitsubishi` | `三菱`, `Mitsubishi Corporation`, `Mitsubishi Shoji` |
| `Reliance` | `Reliance Industries`, `Reliance Industries Limited`, `RIL`, `रिलायंस` |

**Subsidiary / branch / division in the name** (single ORG span when contiguous):

- `Acme Trading (Singapore) Pte. Ltd.`
- `BNP Paribas Hong Kong Branch`
- `Mitsubishi Corporation (Americas)`
- `Acme Trading Asia-Pacific Pte. Ltd.`
- `Citi Private Bank, London`
- `Deutsche Bank AG, Frankfurt`
- `Standard Chartered Bank (Hong Kong) Limited`
- `HSBC India`, `HSBC (UK)`

**Joint-venture / DBA / trade-as patterns:**

- `Acme-XYZ Joint Venture` → single ORG span
- `Acme/XYZ JV` → single ORG span
- `Acme Trading dba Speedy Logistics` → two ORG spans (legal + DBA); document rubric choice
- `Acme Trading t/a Speedy Logistics` → UK "trading as" style
- `Renault-Nissan-Mitsubishi Alliance` → single ORG span (formal alliance)
- `BP-Rosneft JV`

**State-owned / national champion ORGs** (long official + short common form, both labeled):

| Common | Official |
|---|---|
| `China National Petroleum` / `CNPC` | `China National Petroleum Corporation` |
| `Aramco` / `Saudi Aramco` | `Saudi Arabian Oil Company` |
| `Petrobras` | `Petróleo Brasileiro S.A.` |
| `Pemex` | `Petróleos Mexicanos` |
| `ONGC` | `Oil and Natural Gas Corporation Limited` |
| `BHEL` | `Bharat Heavy Electricals Limited` |
| `EDF` | `Électricité de France` |
| `SNCF` | `Société nationale des chemins de fer français` |
| `KEPCO` | `Korea Electric Power Corporation` |
| `Equinor` (post-2018) | `Statoil` (pre-rename) — historical rename pairs are in scope |
| `Meta` (post-2021) | `Facebook, Inc.` |
| `Alphabet` | `Google Inc.` (pre-reorg) |

**Bilingual / dual-script ORG names** (both halves often appear together in source; label the form the source uses, treat as one ORG span when contiguous):

- `Mitsubishi 三菱` → one ORG span if hyphenless and contiguous
- `Samsung 삼성電子` → one ORG span
- `中国移动 China Mobile` → one ORG span
- `HSBC 汇丰银行` → one ORG span
- `LVMH Moët Hennessy Louis Vuitton` → one ORG span

**Fuzzy PERSON variants** (every one of these patterns must appear in dev with the correct gold span):

| Canonical | Variants |
|---|---|
| `Maria Gonzalez` | `MARIA GONZALEZ`, `maria gonzalez`, `MariaGonzalez`, `Maria  Gonzalez`, `Maria Gonzlaez` (typo), `Mria Gonzalez` (dropped char), `Ma1ia Gonzalez` (OCR 1↔r) |
| `Søren Hansen` | `Soren Hansen` (diacritic dropped), `Sören Hansen` (wrong umlaut), `SOREN HANSEN`, `S. Hansen` (initial only) |
| `María Muñoz` | `Maria Munoz`, `MARÍA MUÑOZ`, `MARIA MUNOZ`, `M. Muñoz`, `Maria Munhoz` (typo) |
| `J.K. Rowling` | `JK Rowling`, `J K Rowling`, `J. K. Rowling`, `J.K Rowling`, `J.K.Rowling` (no spaces) |
| `O'Brien` | `OBrien`, `O'Brien` (curly), `O Brien`, `O`Brien` (smart quote) |
| `Mary-Kate` | `Mary–Kate` (en-dash), `Mary—Kate` (em-dash), `MaryKate`, `Mary Kate` |
| `Mao Zedong` | `Mao Tse-tung` (Wade-Giles), `Mao Ze-Dong`, `毛泽东`, `MAO ZEDONG`, `Mao, Zedong` (reversed-with-comma) |
| `Gonzalez, Maria` | reversed CSV order; comma must NOT be inside the span — rubric handles separator |
| `Maria Gonzalez (CEO)` | parenthetical role excluded from span; span = `Maria Gonzalez` |
| `-- Maria Gonzalez` | email-signature artifact; span = `Maria Gonzalez` |
| `Dear Maria,` | salutation form; span = `Maria` |
| `Md. Karim` | abbreviation `Md.` (for Mohammad) **included** in span |
| `A.R. Rahman` | initial-prefix structure; span includes both initials and given |
| `Harpreet Singh` | `Singh` middle/last included as part of span |

**Truncation patterns** (real OCR / API payload artifacts):

- `Maria Gonz…` (ellipsis cut) → span = `Maria Gonz` (truncated text matches; partial entity)
- `Acme Tradi` (mid-word cut) → span = `Acme Tradi` (or rejected by annotator as too ambiguous; document choice)
- `International Business Machines Corp` (last char missing) → span = full text shown
- `42 Industrial Park Road, Rotterda` (city cut mid-word) → ADDRESS span as shown

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
