"""Decoy (non-entity filler) pools for synthetic NER training-data seed generation.

Each key in DECOYS corresponds to exactly one entry in
``scripts.seedgen.ALLOWED_DECOY_SLOTS``.  Values are plain surface strings
inserted verbatim into slot-fill templates; they are *not* labeled as entities.

Spec references:
  §5.2  — negation cue inventory, frozen compounds, contrast/scope cues
           (neg_cue, frozen_compound, contrast_cue reproduce those lists exactly)
  §8.1  — COMMODITY boundary discipline: quantities/units/packaging excluded
           from spans → qty, unit, packaging slots
  §8.3  — PERSON boundary discipline: titles/roles excluded → title, role slots
  §6.x  — record-level composition (multi-entity, tabular, prose)
  §7.x  — noise and artifact types (invoice IDs, dates, currencies, signoffs)
  §10   — zero-entity (true-negative) records → boilerplate slot
  §12   — source mix (invoice, manifest, email, webhook, OCR)
"""

DECOYS: dict[str, list[str]] = {

    # -------------------------------------------------------------------------
    # qty — quantity+unit prefixes that sit BEFORE a commodity and are excluded
    #        from the commodity span (§8.1).  Most end with "of".
    # -------------------------------------------------------------------------
    "qty": [
        "500 tons of",
        "12,000 kg of",
        "300 bales of",
        "40 containers of",
        "two pallets of",
        "1,000 MT of",
        "50 drums of",
        "20 cartons of",
        "15 pallets of",
        "3 FEU of",
        "a shipment of",
        "100 sacks of",
        "5,000 litres of",
        "250 barrels of",
        "6,000 lbs of",
        "80 TEU of",
        "10 reels of",
        "2,500 units of",
        "75 crates of",
        "1 container of",
        "48 rolls of",
        "500 boxes of",
        "30 bags of",
    ],

    # -------------------------------------------------------------------------
    # unit — bare unit tokens (excluded from commodity span per §8.1)
    # -------------------------------------------------------------------------
    "unit": [
        "kg",
        "tons",
        "MT",
        "lbs",
        "g",
        "litres",
        "L",
        "containers",
        "pallets",
        "drums",
        "barrels",
        "TEU",
        "FEU",
        "units",
        "pcs",
    ],

    # -------------------------------------------------------------------------
    # packaging — container nouns used as "<n> <packaging>" and excluded from
    #              the commodity span (§8.1: "300 bales of cotton" → span = cotton)
    # -------------------------------------------------------------------------
    "packaging": [
        "bales of",
        "containers of",
        "drums of",
        "pallets of",
        "barrels of",
        "sacks of",
        "crates of",
        "cartons of",
        "rolls of",
        "reels of",
        "bags of",
        "boxes of",
    ],

    # -------------------------------------------------------------------------
    # invoice_id — purchase-order and invoice reference codes (§12 / §7)
    # -------------------------------------------------------------------------
    "invoice_id": [
        "PO#88231",
        "INV-2024-0421",
        "PO-99812",
        "INV/2024/0912",
        "Invoice #44213",
        "Order 7781-A",
        "PO#20240917-003",
        "INV-2023-9901",
        "PO/2024/5512",
        "Invoice No. 10029",
    ],

    # -------------------------------------------------------------------------
    # ref_id — bill-of-lading, AWB, HS-code, container, and other reference IDs
    # -------------------------------------------------------------------------
    "ref_id": [
        "REF#A-9912",
        "BL-77821",
        "B/L No. 22817",
        "Ref: TX-9981",
        "Container MSKU7782213",
        "AWB 020-44215667",
        "HS 0901.21",
        "Ref: BL-44329",
        "MBL-2024-009981",
        "HBL No. 77-009-2024",
    ],

    # -------------------------------------------------------------------------
    # date — various date/time surface formats (§12 / §7)
    # -------------------------------------------------------------------------
    "date": [
        "2024-09-21",
        "21 Sept 2024",
        "2024-09-21 14:32 UTC",
        "09/21/2024",
        "21/09/2024",
        "Q4 2024",
        "2024-09-21 14:32 UTC — REF#88231",
        "15 Jan 2025",
        "2025-03-07",
        "March 7, 2025",
    ],

    # -------------------------------------------------------------------------
    # currency — currency symbols/codes (§7.5)
    # -------------------------------------------------------------------------
    "currency": [
        "$",
        "€",
        "₹",
        "¥",
        "£",
        "R$",
        "CHF",
        "S$",
        "AED",
        "USD",
        "EUR",
    ],

    # -------------------------------------------------------------------------
    # role — job roles that precede a PERSON and are excluded from the span
    #         (§8.3: "CEO Maria Gonzalez" → span = "Maria Gonzalez")
    # -------------------------------------------------------------------------
    "role": [
        "CEO",
        "Director",
        "Managing Director",
        "buyer",
        "seller",
        "vendor",
        "shipper",
        "consignee",
        "ship-to contact",
        "signed by",
        "approved by",
        "Procurement Lead",
        "Logistics Manager",
        "agent",
    ],

    # -------------------------------------------------------------------------
    # title — honorifics excluded from PERSON span (§4.6 / §8.3)
    # -------------------------------------------------------------------------
    "title": [
        "Mr.",
        "Mrs.",
        "Ms.",
        "Miss",
        "Dr.",
        "Prof.",
        "Mx.",
        "Sir",
        "Dame",
        "Rev.",
        "Fr.",
        "Capt.",
        "Col.",
        "Lt.",
        "Sgt.",
        "Shri",
        "Smt.",
        "Sri",
        "Pandit",
        "Don",
        "Doña",
        "Herr",
        "Frau",
        "Sheikh",
        "Hajji",
        "Sayyid",
    ],

    # -------------------------------------------------------------------------
    # salutation — address-form prefixes excluded from PERSON span (§4.6)
    # -------------------------------------------------------------------------
    "salutation": [
        "Dear",
        "Hi",
        "Hello",
        "Attn:",
        "To:",
        "c/o",
        "Attention:",
    ],

    # -------------------------------------------------------------------------
    # signoff — email/letter closing fragments (§10 / §7)
    # -------------------------------------------------------------------------
    "signoff": [
        "Best regards,",
        "Regards,",
        "Kind regards,",
        "Thanks,",
        "Many thanks,",
        "Sincerely,",
        "Sent from my iPhone",
        "Sent from my mobile",
        "--",
        "Best,",
    ],

    # -------------------------------------------------------------------------
    # boilerplate — complete zero-entity sentences/fragments for true-negative
    #               templates (§10).  Must contain NO person/org/address/commodity.
    # -------------------------------------------------------------------------
    "boilerplate": [
        "This is an automated message. Do not reply.",
        "The shipment arrived on time.",
        "Please find the details below.",
        "Thank you for your business.",
        "Kindly confirm receipt of this email.",
        "All prices are subject to change without notice.",
        "This email and any attachments are confidential.",
        "Delivery is expected within 5 business days.",
        "Please do not hesitate to contact us.",
        "Terms and conditions apply.",
        "The goods were inspected and found satisfactory.",
        "Awaiting your confirmation.",
        "See attached for further information.",
        "This message was sent automatically. Please do not reply.",
        "Payment is due within 30 days of invoice date.",
    ],

    # -------------------------------------------------------------------------
    # neg_cue — PRESERVED real negation cues (§5.2); these trigger polarity=NEG
    #            when in scope and must NOT be mutated by the noise injector
    #            (Record.meta["preserve_spans"] covers their char ranges).
    # -------------------------------------------------------------------------
    "neg_cue": [
        "no",
        "not",
        "none",
        "never",
        "does not contain",
        "do not contain",
        "cannot contain",
        "without",
        "lacks",
        "lacking",
        "excludes",
        "excluding",
        "free of",
        "absent of",
        "missing",
        "omitted",
        "unauthorized",
        "forbidden",
        "prohibited",
        "cannot include",
    ],

    # -------------------------------------------------------------------------
    # contrast_cue — PRESERVED contrast/scope boundary markers (§5.2);
    #                each begins with the punctuation shown in the spec.
    # -------------------------------------------------------------------------
    "contrast_cue": [
        ", but only",
        ", only",
        ", instead",
        ", rather than",
        "; instead of",
        "; in lieu of",
        ", except",
        ", excluding",
    ],

    # -------------------------------------------------------------------------
    # frozen_compound — dissolved negation-looking cues that must NOT trigger
    #                    polarity=NEG; commodity spans including these are POS
    #                    (§5.2 frozen-compound list, reproduced exactly).
    # -------------------------------------------------------------------------
    "frozen_compound": [
        "sugar-free",
        "gluten-free",
        "lead-free",
        "BPA-free",
        "oil-free",
        "fat-free",
        "dairy-free",
        "caffeine-free",
        "tax-free",
        "duty-free",
        "non-stick",
        "non-toxic",
        "non-flammable",
        "non-allergenic",
        "stainless",
        "seamless",
        "careless",
        "noise-cancelling",
        "wrinkle-resistant",
    ],

    # -------------------------------------------------------------------------
    # incoterm — trade-terms codes (§12 / source mix: invoices, manifests)
    # -------------------------------------------------------------------------
    "incoterm": [
        "FOB",
        "CIF",
        "CFR",
        "EXW",
        "DDP",
        "DAP",
        "FCA",
        "CIP",
    ],
}
