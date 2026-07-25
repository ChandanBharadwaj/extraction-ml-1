"""Hand-labeled gold validation records.

These are the *sole* source of truth for early stopping and metric reporting.
Synthetic data should never bleed into this set. The seed examples below cover
the canonical edge cases from the TDD plus a curated set of negation records
that exercise the four hard properties of negation handling:

    Scope          — how far the negation reaches across coordinated commodities.
    Target spec.   — denial sticks to a qualifier ("special wood") while the
                     bare form may still be positively asserted in the same record.
    Dissolved cues — "sugar-free", "stainless", "non-stick" are NOT negations
                     in this domain even though they contain negation-looking
                     surface forms.
    Word order     — denied and asserted entities sit on either side of the cue.

Replace / extend this list as your hand-labeling effort grows. Aim for
500-2,000 examples before production training.
"""
from __future__ import annotations

from ner.schema import Entity, Record


def _r(text: str, *entities: tuple[str, str, int, int]) -> Record:
    """TDD-style 4-tuple constructor (type, text, start, end) — POS polarity."""
    rec = Record(text=text, entities=[Entity(t, s, a, b) for t, s, a, b in entities])
    rec.validate()
    return rec


def _n(text: str, *entities: tuple[str, str, int, int, str]) -> Record:
    """Negation-aware 5-tuple constructor (type, text, start, end, polarity)."""
    rec = Record(text=text, entities=[
        Entity(type=t, text=s, start=a, end=b, polarity=p)
        for t, s, a, b, p in entities
    ])
    rec.validate()
    return rec


# 5 canonical TDD examples (positive-only).
TDD_SEED: list[Record] = [
    _r(
        "Invoice: 500 tons of Grade A robusta coffee for Nordwind Logistics GmbH.",
        ("COMMODITY", "Grade A robusta coffee", 21, 43),
        ("ORG", "Nordwind Logistics GmbH", 48, 71),
    ),
    _r(
        "shipment of 304 stainless steel sheet approved by maria gonzalez at acme trading co",
        ("COMMODITY", "304 stainless steel sheet", 12, 37),
        ("PERSON", "maria gonzalez", 50, 64),
        ("ORG", "acme trading co", 68, 83),
    ),
    _r(
        "PO#88231 | Polyethylene resin HDPE | Qty 12,000 kg | Sold to: Delta Packaging, 7 Canal St, Singapore 049320",
        ("COMMODITY", "Polyethylene resin HDPE", 11, 34),
        ("ORG", "Delta Packaging", 62, 77),
        ("ADDRESS", "7 Canal St, Singapore 049320", 79, 107),
    ),
    _r(
        "Maria Gonzalez from Acme Trading Co. confirmed refined copper cathode shipped to 42 Industrial Park Road, Rotterdam, 3011 AB.",
        ("PERSON", "Maria Gonzalez", 0, 14),
        ("ORG", "Acme Trading Co.", 20, 36),
        ("COMMODITY", "refined copper cathode", 47, 69),
        ("ADDRESS", "42 Industrial Park Road, Rotterdam, 3011 AB", 81, 124),
    ),
    _r(
        "Manifest: galvanized steel coil, anhydrous ammonia, raw cane sugar — ETA Felix Yu, Oceanic Freight Co.",
        ("COMMODITY", "galvanized steel coil", 10, 31),
        ("COMMODITY", "anhydrous ammonia", 33, 50),
        ("COMMODITY", "raw cane sugar", 52, 66),
        ("PERSON", "Felix Yu", 73, 81),
        ("ORG", "Oceanic Freight Co.", 83, 102),
    ),
]


# 30 negation-aware records covering scope / target specificity / dissolved
# cues / word order. Offsets verified by find() over the source string.
NEGATION_SEED: list[Record] = [
    _n('Manifest does not contain wood, lead-free paint, or treated wood.',
        ('COMMODITY', 'wood', 26, 30, 'NEG'),
        ('COMMODITY', 'treated wood', 52, 64, 'NEG'),
    ),
    _n('Acme Trading Co. certifies no asbestos, lead, or mercury in this shipment.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'asbestos', 30, 38, 'NEG'),
        ('COMMODITY', 'lead', 40, 44, 'NEG'),
        ('COMMODITY', 'mercury', 49, 56, 'NEG'),
    ),
    _n('PO#88231: does not contain raw cane sugar, cotton, or anhydrous ammonia.',
        ('COMMODITY', 'raw cane sugar', 27, 41, 'NEG'),
        ('COMMODITY', 'cotton', 43, 49, 'NEG'),
        ('COMMODITY', 'anhydrous ammonia', 54, 71, 'NEG'),
    ),
    _n('Shipment contains no special wood, but ordinary wood is acceptable.',
        ('COMMODITY', 'special wood', 21, 33, 'NEG'),
        ('COMMODITY', 'wood', 48, 52, 'POS'),
    ),
    _n('No Grade A robusta coffee; standard robusta coffee shipped instead.',
        ('COMMODITY', 'Grade A robusta coffee', 3, 25, 'NEG'),
        ('COMMODITY', 'robusta coffee', 36, 50, 'POS'),
    ),
    _n('Delivery excludes 304 stainless steel sheet; plain steel sheet is fine.',
        ('COMMODITY', '304 stainless steel sheet', 18, 43, 'NEG'),
        ('COMMODITY', 'steel sheet', 51, 62, 'POS'),
    ),
    _n('Order rejects treated wood, accepts wood.',
        ('COMMODITY', 'treated wood', 14, 26, 'NEG'),
        ('COMMODITY', 'wood', 36, 40, 'POS'),
    ),
    _n('Shipment of sugar-free chocolate and stainless steel sheet to Delta Packaging.',
        ('COMMODITY', 'sugar-free chocolate', 12, 32, 'POS'),
        ('COMMODITY', 'stainless steel sheet', 37, 58, 'POS'),
        ('ORG', 'Delta Packaging', 62, 77, 'POS'),
    ),
    _n('BlueRiver Commodities Ltd supplies non-stick cookware coatings.',
        ('ORG', 'BlueRiver Commodities Ltd', 0, 25, 'POS'),
        ('COMMODITY', 'non-stick cookware coatings', 35, 62, 'POS'),
    ),
    _n('Felix Yu confirmed gluten-free flour delivery.',
        ('PERSON', 'Felix Yu', 0, 8, 'POS'),
        ('COMMODITY', 'gluten-free flour', 19, 36, 'POS'),
    ),
    _n('Delivery of lead-free solder approved by Maria Gonzalez.',
        ('COMMODITY', 'lead-free solder', 12, 28, 'POS'),
        ('PERSON', 'Maria Gonzalez', 41, 55, 'POS'),
    ),
    _n('Does not contain copper cathode, only refined copper cathode shipped.',
        ('COMMODITY', 'copper cathode', 17, 31, 'NEG'),
        ('COMMODITY', 'refined copper cathode', 38, 60, 'POS'),
    ),
    _n('Refined copper cathode shipped, but no copper cathode reserves remaining.',
        ('COMMODITY', 'Refined copper cathode', 0, 22, 'POS'),
        ('COMMODITY', 'copper cathode', 39, 53, 'NEG'),
    ),
    _n('Cotton confirmed; organic cotton not available.',
        ('COMMODITY', 'Cotton', 0, 6, 'POS'),
        ('COMMODITY', 'organic cotton', 18, 32, 'NEG'),
    ),
    _n('No anhydrous ammonia in stock; ammonia substitutes acceptable.',
        ('COMMODITY', 'anhydrous ammonia', 3, 20, 'NEG'),
        ('COMMODITY', 'ammonia', 31, 38, 'POS'),
    ),
    _n('Acme Trading Co. shipment lacks raw cane sugar.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'raw cane sugar', 32, 46, 'NEG'),
    ),
    _n('This delivery is free of asbestos.',
        ('COMMODITY', 'asbestos', 25, 33, 'NEG'),
    ),
    _n('Without Polyethylene resin HDPE in the manifest.',
        ('COMMODITY', 'Polyethylene resin HDPE', 8, 31, 'NEG'),
    ),
    _n('Container absent of galvanized steel coil.',
        ('COMMODITY', 'galvanized steel coil', 20, 41, 'NEG'),
    ),
    _n('Maria Gonzalez at Acme Trading Co. confirmed: no special wood, only treated wood.',
        ('PERSON', 'Maria Gonzalez', 0, 14, 'POS'),
        ('ORG', 'Acme Trading Co.', 18, 34, 'POS'),
        ('COMMODITY', 'special wood', 49, 61, 'NEG'),
        ('COMMODITY', 'treated wood', 68, 80, 'POS'),
    ),
    _n('Felix Yu reports Oceanic Freight Co. did not deliver refined copper cathode.',
        ('PERSON', 'Felix Yu', 0, 8, 'POS'),
        ('ORG', 'Oceanic Freight Co.', 17, 36, 'POS'),
        ('COMMODITY', 'refined copper cathode', 53, 75, 'NEG'),
    ),
    _n('Nordwind Logistics GmbH excludes hazardous materials: no anhydrous ammonia, no mercury.',
        ('ORG', 'Nordwind Logistics GmbH', 0, 23, 'POS'),
        ('COMMODITY', 'anhydrous ammonia', 57, 74, 'NEG'),
        ('COMMODITY', 'mercury', 79, 86, 'NEG'),
    ),
    _n('shipment does not contain wood, only plastic toys',
        ('COMMODITY', 'wood', 26, 30, 'NEG'),
        ('COMMODITY', 'plastic toys', 37, 49, 'POS'),
    ),
    _n('no robusta coffee in this batch; arabica beans only.',
        ('COMMODITY', 'robusta coffee', 3, 17, 'NEG'),
        ('COMMODITY', 'arabica beans', 33, 46, 'POS'),
    ),
    _n('Sugar-free chocolate confirmed; no raw cane sugar in any product.',
        ('COMMODITY', 'Sugar-free chocolate', 0, 20, 'POS'),
        ('COMMODITY', 'raw cane sugar', 35, 49, 'NEG'),
    ),
    _n('Stainless steel sheet shipped, but no galvanized steel coil this run.',
        ('COMMODITY', 'Stainless steel sheet', 0, 21, 'POS'),
        ('COMMODITY', 'galvanized steel coil', 38, 59, 'NEG'),
    ),
    _n('Delivery to 7 Canal St, Singapore 049320 does not include cotton.',
        ('ADDRESS', '7 Canal St, Singapore 049320', 12, 40, 'POS'),
        ('COMMODITY', 'cotton', 58, 64, 'NEG'),
    ),
    _n('Shipment to 42 Industrial Park Road, Rotterdam, 3011 AB excludes lead-free solder containers.',
        ('ADDRESS', '42 Industrial Park Road, Rotterdam, 3011 AB', 12, 55, 'POS'),
        ('COMMODITY', 'lead-free solder containers', 65, 92, 'NEG'),
    ),
    _n('Acme Trading Co. ships robusta coffee but lacks Grade A robusta coffee at present.',
        ('ORG', 'Acme Trading Co.', 0, 16, 'POS'),
        ('COMMODITY', 'robusta coffee', 23, 37, 'POS'),
        ('COMMODITY', 'Grade A robusta coffee', 48, 70, 'NEG'),
    ),
    _n('Kenji Tanaka noted: no treated wood, no special wood, no plywood in the order.',
        ('PERSON', 'Kenji Tanaka', 0, 12, 'POS'),
        ('COMMODITY', 'treated wood', 23, 35, 'NEG'),
        ('COMMODITY', 'special wood', 40, 52, 'NEG'),
        ('COMMODITY', 'plywood', 57, 64, 'NEG'),
    ),
]


# Hand-authored cargo-document records: bills of lading, manifests, packing
# lists, customs declarations — the registers production traffic actually
# speaks. ALL-CAPS/telex casing, STC/container/seal/vessel/port furniture
# (unlabeled), HS-heading phrasing with frozen "NOT", XL trade-spec spans,
# NIL/no-cargo-other-than denials, bare-quantity boundaries, one multi-line
# packing list, and one >500-char full B/L body. Offsets computed by Python
# find() and enforced by Record.validate() — never slot-fill output
# (synthetic data must not leak into eval).
CARGO_SEED: list[Record] = [
    _r('DESCRIPTION OF GOODS: WHITE REFINED CANE SUGAR ICUMSA 45 PACKED IN 50KG PP BAGS',
        ('COMMODITY', 'WHITE REFINED CANE SUGAR ICUMSA 45', 22, 56),
    ),
    _r('1X40HC STC 500 BAGS RAW ARABICA COFFEE BEANS ORIGIN COLOMBIA',
        ('COMMODITY', 'RAW ARABICA COFFEE BEANS', 20, 44),
    ),
    _r('CARGO: HOT-ROLLED STEEL COIL GROSS 24,500 KGS FREIGHT PREPAID',
        ('COMMODITY', 'HOT-ROLLED STEEL COIL', 7, 28),
    ),
    _r('SAID TO CONTAIN GENERAL CARGO AND USED HOUSEHOLD GOODS',
        ('COMMODITY', 'GENERAL CARGO', 16, 29),
        ('COMMODITY', 'USED HOUSEHOLD GOODS', 34, 54),
    ),
    _r('MV NORDIC STAR V.023E | POL SGSIN POD NLRTM | STC: COPPER CATHODES | SEAL NO. CN1234567',
        ('COMMODITY', 'COPPER CATHODES', 51, 66),
    ),
    _r('VESSEL MV EVER GIVEN VOY 118W CARRIER MAEU CARGO SOYBEAN MEAL IN BULK',
        ('COMMODITY', 'SOYBEAN MEAL', 49, 61),
    ),
    _r('BOOKING CONF: 2X40HC EX PORT KLANG TO ROTTERDAM, CARGO PET RESIN, SHIPPER PACIFIC POLYMERS SDN BHD',
        ('COMMODITY', 'PET RESIN', 55, 64),
        ('ORG', 'PACIFIC POLYMERS SDN BHD', 74, 98),
    ),
    _r('HS 0901.21 ROASTED COFFEE, NOT DECAFFEINATED | NET 12,000 KG',
        ('COMMODITY', 'ROASTED COFFEE, NOT DECAFFEINATED', 11, 44),
    ),
    _r('CUSTOMS ENTRY HS 1701.14: RAW CANE SUGAR, IN SOLID FORM | 500 MT',
        ('COMMODITY', 'RAW CANE SUGAR, IN SOLID FORM', 26, 55),
    ),
    _r('hs code 7208.39 hot-rolled coil of non-alloy steel per attached mill cert',
        ('COMMODITY', 'hot-rolled coil of non-alloy steel', 16, 50),
    ),
    _r('OFFER: FOOD-GRADE REFINED BLEACHED DEODORISED PALM OLEIN RBD CP8 IODINE VALUE 60 MINIMUM CIF NHAVA SHEVA',
        ('COMMODITY', 'FOOD-GRADE REFINED BLEACHED DEODORISED PALM OLEIN RBD CP8 IODINE VALUE 60 MINIMUM', 7, 88),
    ),
    _r('Cargo manifest lists high-density polyethylene injection-moulding grade resin HDPE 5502 natural pellets conforming to ASTM D4703, loaded at Jebel Ali.',
        ('COMMODITY', 'high-density polyethylene injection-moulding grade resin HDPE 5502 natural pellets conforming to ASTM D4703', 21, 128),
    ),
    _r('SPEC: GRANULAR UREA 46 PERCENT NITROGEN MINIMUM TREATED WITH ANTI-CAKING AGENT IN JUMBO BAGS',
        ('COMMODITY', 'GRANULAR UREA 46 PERCENT NITROGEN MINIMUM TREATED WITH ANTI-CAKING AGENT', 6, 78),
    ),
    _n('NIL DANGEROUS GOODS DECLARED. CARGO: GENERAL MERCHANDISE ONLY.',
        ('COMMODITY', 'DANGEROUS GOODS', 4, 19, 'NEG'),
        ('COMMODITY', 'GENERAL MERCHANDISE', 37, 56, 'POS'),
    ),
    _n('Certificate confirms shipment is FREE OF ASBESTOS; cement clinker only.',
        ('COMMODITY', 'ASBESTOS', 41, 49, 'NEG'),
        ('COMMODITY', 'cement clinker', 51, 65, 'POS'),
    ),
    _r('No cargo other than skimmed milk powder is stowed in container MSKU7782213.',
        ('COMMODITY', 'skimmed milk powder', 20, 39),
    ),
    _n('MANIFEST DECLARES NO LITHIUM ION BATTERIES; SPARE PARTS AND USED MACHINERY ONLY.',
        ('COMMODITY', 'LITHIUM ION BATTERIES', 21, 42, 'NEG'),
        ('COMMODITY', 'SPARE PARTS', 44, 55, 'POS'),
        ('COMMODITY', 'USED MACHINERY', 60, 74, 'POS'),
    ),
    _n('shipper certifies consignment contains no unrefined shea butter, only refined shea butter.',
        ('COMMODITY', 'unrefined shea butter', 42, 63, 'NEG'),
        ('COMMODITY', 'refined shea butter', 70, 89, 'POS'),
    ),
    _n('cargo does not include arms, ammunition, or fireworks per shipper declaration',
        ('COMMODITY', 'arms', 23, 27, 'NEG'),
        ('COMMODITY', 'ammunition', 29, 39, 'NEG'),
        ('COMMODITY', 'fireworks', 44, 53, 'NEG'),
    ),
    _n('TELEX: NO CRUDE PALM OIL ON BOARD, RBD PALM OLEIN ONLY',
        ('COMMODITY', 'CRUDE PALM OIL', 10, 24, 'NEG'),
        ('COMMODITY', 'RBD PALM OLEIN', 35, 49, 'POS'),
    ),
    _n('Container holds no general cargo; SULPHURIC ACID UN 1830 only.',
        ('COMMODITY', 'general cargo', 19, 32, 'NEG'),
        ('COMMODITY', 'SULPHURIC ACID UN 1830', 34, 56, 'POS'),
    ),
    _n('Nil personal effects declared; consignment is auto spare parts from Tata International Ltd.',
        ('COMMODITY', 'personal effects', 4, 20, 'NEG'),
        ('COMMODITY', 'auto spare parts', 46, 62, 'POS'),
        ('ORG', 'Tata International Ltd.', 68, 91, 'POS'),
    ),
    _n('WITHOUT PET RESIN OR PVC RESIN — CARGO IS KRAFT LINER BOARD 175GSM',
        ('COMMODITY', 'PET RESIN', 8, 17, 'NEG'),
        ('COMMODITY', 'PVC RESIN', 21, 30, 'NEG'),
        ('COMMODITY', 'KRAFT LINER BOARD 175GSM', 42, 66, 'POS'),
    ),
    _r('12,000 KG HDPE RESIN LOADED AT LAEM CHABANG',
        ('COMMODITY', 'HDPE RESIN', 10, 20),
    ),
    _r('Packing: 500 bags basmati rice, 40 pallets wheat flour, 1 lot spare parts.',
        ('COMMODITY', 'basmati rice', 18, 30),
        ('COMMODITY', 'wheat flour', 43, 54),
        ('COMMODITY', 'spare parts', 62, 73),
    ),
    _r('NET 24,000 KGS COLD-ROLLED STEEL COIL / GROSS 24,850 KGS',
        ('COMMODITY', 'COLD-ROLLED STEEL COIL', 15, 37),
    ),
    _r('Deliver used agricultural machinery to 88 Harbour Road, Mombasa 80100, attn Grace Wanjiku.',
        ('COMMODITY', 'used agricultural machinery', 8, 35),
        ('ADDRESS', '88 Harbour Road, Mombasa 80100', 39, 69),
        ('PERSON', 'Grace Wanjiku', 76, 89),
    ),
    _r('CONSIGNEE: DELTA PACKAGING, 7 CANAL ST, SINGAPORE 049320 | CARGO: A4 COPY PAPER 80GSM',
        ('ORG', 'DELTA PACKAGING', 11, 26),
        ('ADDRESS', '7 CANAL ST, SINGAPORE 049320', 28, 56),
        ('COMMODITY', 'A4 COPY PAPER 80GSM', 66, 85),
    ),
    _r('Ship 2,000 drums isopropanol UN 1219 to 15 Rue de la Gare, 69002 Lyon.',
        ('COMMODITY', 'isopropanol UN 1219', 17, 36),
        ('ADDRESS', '15 Rue de la Gare, 69002 Lyon', 40, 69),
    ),
    _r('Invoice covers 1,200 cases Nescafé instant coffee and 300 cases Häagen-Dazs ice cream.',
        ('COMMODITY', 'Nescafé instant coffee', 27, 49),
        ('COMMODITY', 'Häagen-Dazs ice cream', 64, 85),
    ),
    _r('MARKS: N/M | ITEM: MICHELIN RADIAL TRUCK TYRES QTY 100',
        ('COMMODITY', 'MICHELIN RADIAL TRUCK TYRES', 19, 46),
    ),
    _r('TALLY: C0PPER CATH0DES 25 MT SEAL 0451227',
        ('COMMODITY', 'C0PPER CATH0DES', 7, 22),
    ),
    _r('we offer cpo 500 mt cif rotterdam and meg 200 mt prompt shipment',
        ('COMMODITY', 'cpo', 9, 12),
        ('COMMODITY', 'meg', 38, 41),
    ),
    _r('PACKING LIST — INV-2024-0788\n1. 500 CTNS FROZEN SHRIMP, HEADLESS SHELL-ON\n2. 250 CTNS FROZEN FILLETS OF ALASKA POLLACK\n3. 100 CTNS FROZEN CHICKEN\nNOTIFY: ARCTIC FOODS BV, HAVENSTRAAT 12, 3011 AB ROTTERDAM',
        ('COMMODITY', 'FROZEN SHRIMP, HEADLESS SHELL-ON', 41, 73),
        ('COMMODITY', 'FROZEN FILLETS OF ALASKA POLLACK', 86, 118),
        ('COMMODITY', 'FROZEN CHICKEN', 131, 145),
        ('ORG', 'ARCTIC FOODS BV', 154, 169),
        ('ADDRESS', 'HAVENSTRAAT 12, 3011 AB ROTTERDAM', 171, 204),
    ),
    _n('BILL OF LADING BL-2024-88231\nSHIPPER: GOLDEN HARVEST AGRI EXPORTS PTE LTD\nCONSIGNEE: NORDWIND LOGISTICS GMBH, LAGERSTRASSE 44, 20095 HAMBURG\nVESSEL: MV PACIFIC HARMONY V.118W  POL: SGSIN  POD: DEHAM\n1X40HC STC 760 BAGS WHITE REFINED CANE SUGAR ICUMSA 45 POLARISATION 99.80 DEGREES MINIMUM\n1X20GP STC 500 BAGS LONG GRAIN WHITE RICE, SEMI-MILLED, WHETHER OR NOT POLISHED\nSEAL NO. CN1234567 / SEAL NO: HLA2214467\nGROSS WEIGHT 44,250.00 KGS  NET WEIGHT 43,800.00 KGS\nFREIGHT PREPAID  SHIPPED ON BOARD 21 SEPT 2025\nNIL DANGEROUS GOODS DECLARED',
        ('ORG', 'GOLDEN HARVEST AGRI EXPORTS PTE LTD', 38, 73, 'POS'),
        ('ORG', 'NORDWIND LOGISTICS GMBH', 85, 108, 'POS'),
        ('ADDRESS', 'LAGERSTRASSE 44, 20095 HAMBURG', 110, 140, 'POS'),
        ('COMMODITY', 'WHITE REFINED CANE SUGAR ICUMSA 45 POLARISATION 99.80 DEGREES MINIMUM', 219, 288, 'POS'),
        ('COMMODITY', 'LONG GRAIN WHITE RICE, SEMI-MILLED, WHETHER OR NOT POLISHED', 309, 368, 'POS'),
        ('COMMODITY', 'DANGEROUS GOODS', 514, 529, 'NEG'),
    ),
    _n('B/L REMARK: NO SCRAP METAL, NO USED TYRES — CARGO IS CEMENT CLINKER IN BULK',
        ('COMMODITY', 'SCRAP METAL', 15, 26, 'NEG'),
        ('COMMODITY', 'USED TYRES', 31, 41, 'NEG'),
        ('COMMODITY', 'CEMENT CLINKER', 53, 67, 'POS'),
    ),
    _n('surveyor found no ICUMSA 45 sugar on board, only raw sugar ICUMSA 600-1200',
        ('COMMODITY', 'ICUMSA 45 sugar', 18, 33, 'NEG'),
        ('COMMODITY', 'raw sugar ICUMSA 600-1200', 49, 74, 'POS'),
    ),
]


GOLD_SEED: list[Record] = [*TDD_SEED, *NEGATION_SEED, *CARGO_SEED]


def load_gold(path: str | None = None) -> list[Record]:
    """Load gold records: from `path` if provided, else the in-memory seed set."""
    if path is None:
        return list(GOLD_SEED)
    from ner.data.assembler import read_jsonl
    return read_jsonl(path)


GOLD_SPLITS: tuple[str, ...] = ("all", "earlystop", "tune")


def split_gold(records: list[Record], split: str) -> list[Record]:
    """Deterministically partition gold so early stopping ("earlystop") and
    threshold tuning ("tune") don't consume identical records.

    Assignment hashes the record text (sha1 % 2), so it is stable across
    runs, record ordering, and gold-set growth — adding records never moves
    existing ones between halves. "all" returns everything unchanged.
    """
    if split == "all":
        return list(records)
    if split not in GOLD_SPLITS:
        raise ValueError(f"unknown gold split {split!r}; expected one of {GOLD_SPLITS}")
    import hashlib

    want = 0 if split == "earlystop" else 1
    return [
        r for r in records
        if int(hashlib.sha1(r.text.encode("utf-8")).hexdigest(), 16) % 2 == want
    ]
