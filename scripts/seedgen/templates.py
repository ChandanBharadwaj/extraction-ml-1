"""Slot-fill template library for synthetic NER training data.

Grammar reference: ner/data/slot_fill.py
Spec sections covered: 5 (negation), 6 (record composition), 7.4 (tabular),
8 (boundary discipline), 10 (zero-entity), 12 (source mix).

Slot token rules:
  - Entity slots: {PERSON} {ORG} {ADDRESS} {COMMODITY} {NEG_COMMODITY}
  - Indexed forms: {PERSON#1} {PERSON#2} {COMMODITY#1} {NEG_COMMODITY#1} etc.
    Required whenever the SAME type appears more than once in one template.
  - Decoy slots: {decoy:X} where X is one of the ALLOWED_DECOY_SLOTS from
    scripts/seedgen/__init__.py — qty, unit, packaging, invoice_id, ref_id,
    date, currency, role, title, salutation, signoff, boilerplate, neg_cue,
    contrast_cue, frozen_compound, incoterm.
  - {decoy:neg_cue} and {decoy:contrast_cue} are PRESERVED (noise cannot
    alter them) so polarity labels stay correct.
"""

from __future__ import annotations

TEMPLATES: list[str] = [

    # -------------------------------------------------------------------------
    # A. INVOICE HEADERS — tabular, pipe/tab-delimited, PO/date/currency lines
    # (§12: invoice style, ~35% of production traffic)
    # -------------------------------------------------------------------------

    # A-1: Basic invoice header (4 entities: all types)
    "{PERSON} from {ORG} confirmed {COMMODITY} shipped to {ADDRESS}.",

    # A-2: Pipe-delimited invoice line with qty and currency
    "{decoy:invoice_id} | {decoy:qty} {COMMODITY} | {decoy:currency}{decoy:qty} | {ORG}",

    # A-3: Tab-delimited invoice row
    "{decoy:invoice_id}\t{COMMODITY}\t{decoy:qty}\t{ORG}\t{ADDRESS}",

    # A-4: Invoice with date and incoterm
    "Invoice {decoy:invoice_id} dated {decoy:date}: {decoy:qty} {COMMODITY}, {decoy:incoterm}, shipped by {ORG} to {ADDRESS}.",

    # A-5: PO confirmation with person and currency
    "{decoy:invoice_id} — {decoy:title} {PERSON} ({ORG}) confirmed {decoy:currency}{decoy:qty} for {COMMODITY}.",

    # A-6: Key:value style invoice fields
    "Buyer: {ORG} | Ship-to: {ADDRESS} | Commodity: {COMMODITY} | Ref: {decoy:ref_id}",

    # A-7: Invoice with two commodities
    "{decoy:invoice_id}: {COMMODITY#1} and {COMMODITY#2} invoiced to {ORG}.",

    # A-8: Multi-line invoice header (embedded newlines)
    "PO: {decoy:invoice_id}\nSeller: {ORG}\nBuyer: {PERSON}\nGoods: {COMMODITY}",

    # A-9: Invoice with incoterm and address
    "{decoy:incoterm} {ADDRESS} — {ORG} dispatches {decoy:qty} {COMMODITY} per {decoy:invoice_id}.",

    # A-10: Full invoice line with currency, qty, all four types
    "{decoy:invoice_id} | Seller: {ORG} | Buyer: {PERSON} | {decoy:currency}{decoy:qty} {COMMODITY} | Destination: {ADDRESS}",

    # A-11: Tab-delimited with ref and date
    "REF\t{decoy:ref_id}\tDate\t{decoy:date}\tSupplier\t{ORG}\tItem\t{COMMODITY}",

    # A-12: Invoice with two orgs (seller + buyer)
    "{decoy:invoice_id}: sold by {ORG#1} to {ORG#2}, goods: {COMMODITY}, terms {decoy:incoterm}.",

    # A-13: Currency amount + commodity + person signer
    "{decoy:currency}{decoy:qty} invoiced for {COMMODITY}; signed by {decoy:title} {PERSON} on {decoy:date}.",

    # A-14: Short invoice stub (2 entities)
    "Invoice {decoy:invoice_id}: {COMMODITY} for {ORG}.",

    # A-15: Pipe-delimited with person, role excluded
    "{decoy:role} {PERSON} | {ORG} | {COMMODITY} | {decoy:date}",

    # -------------------------------------------------------------------------
    # B. MANIFESTS / PACKING LISTS — serial commodity lists, addresses, newlines
    # (§12: manifest style, ~25% of traffic)
    # -------------------------------------------------------------------------

    # B-1: Classic manifest with three commodities
    "Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} — ETA {PERSON}, {ORG}.",

    # B-2: Packing list with address, multiline
    "Packing List\nConsignee: {ORG}\nDeliver to: {ADDRESS}\nItems: {COMMODITY#1}, {COMMODITY#2}",

    # B-3: Serial commodity manifest with qty and incoterm
    "Manifest {decoy:ref_id}: {decoy:qty} {COMMODITY#1}; {decoy:qty} {COMMODITY#2}; {decoy:incoterm} {ADDRESS}.",

    # B-4: Five commodities (5+ entity bucket)
    "Cargo manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}, {COMMODITY#4}, {COMMODITY#5} consigned to {ORG}.",

    # B-5: Manifest with person and org (ETA signature)
    "Bill of Lading {decoy:ref_id}: {ORG} ships {COMMODITY} to {ADDRESS}, contact {PERSON}.",

    # B-6: Tabular manifest (pipe-delimited, multi-commodity)
    "{COMMODITY#1} | {decoy:qty} | {decoy:incoterm}\n{COMMODITY#2} | {decoy:qty} | {decoy:incoterm}\nConsignee: {ORG}",

    # B-7: Manifest with packaging decoy
    "Shipped: {decoy:packaging} {COMMODITY#1} and {decoy:packaging} {COMMODITY#2} to {ADDRESS}.",

    # B-8: Long manifest with all four types
    "Manifest {decoy:invoice_id}: {decoy:qty} {COMMODITY#1}, {decoy:qty} {COMMODITY#2} loaded by {ORG}, consigned to {PERSON} at {ADDRESS}.",

    # B-9: Two-address manifest (multi-address coverage §6.4)
    "Ship {COMMODITY#1} to {ADDRESS#1}; ship {COMMODITY#2} to {ADDRESS#2}.",

    # B-10: Manifest with currency
    "{ORG} manifest {decoy:ref_id}: {COMMODITY} valued at {decoy:currency}{decoy:qty}, delivered {decoy:incoterm} to {ADDRESS}.",

    # B-11: Multiline packing list with tab separators
    "Line 1:\t{COMMODITY#1}\t{decoy:qty}\nLine 2:\t{COMMODITY#2}\t{decoy:qty}\nShip-to:\t{ADDRESS}",

    # B-12: Manifest with four commodities
    "{ORG} declares: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}, {COMMODITY#4} per {decoy:invoice_id}.",

    # -------------------------------------------------------------------------
    # C. EMAILS — prose, salutation + signoff, multi-party
    # (§12: email style, ~20% of traffic)
    # -------------------------------------------------------------------------

    # C-1: Basic email body
    "{decoy:salutation} {PERSON}, please confirm shipment of {COMMODITY} from {ORG} to {ADDRESS}. {decoy:signoff}",

    # C-2: Email with two persons
    "{decoy:salutation} {PERSON#1}, please coordinate with {PERSON#2} regarding {COMMODITY} delivery.",

    # C-3: Email body with org and commodity
    "We at {ORG} are pleased to confirm the dispatch of {decoy:qty} {COMMODITY} per your order {decoy:invoice_id}.",

    # C-4: Email with signoff person
    "The consignment of {COMMODITY} is ready for pickup at {ADDRESS}. Kindly advise. {decoy:signoff} {PERSON}",

    # C-5: Email with two orgs
    "{decoy:salutation} {PERSON}, {ORG#1} has approved the transfer to {ORG#2} for {decoy:qty} {COMMODITY}.",

    # C-6: Email thread style with role
    "{decoy:role} {PERSON} at {ORG} confirmed: shipment to {ADDRESS} will include {COMMODITY}.",

    # C-7: Email footer / signature style
    "Regards to {ORG} — we await delivery of {COMMODITY}. {decoy:signoff} {decoy:title} {PERSON}",

    # C-8: Email with date and ref
    "As of {decoy:date}, {decoy:ref_id}: {ORG} has dispatched {COMMODITY} to {ADDRESS}.",

    # C-9: Long prose email body
    "{decoy:salutation} {PERSON}, this is to confirm that {ORG} will ship {decoy:qty} {COMMODITY} under {decoy:incoterm} terms to {ADDRESS} on {decoy:date}.",

    # C-10: Short informal email
    "Hi {PERSON}, can you check with {ORG} about the {COMMODITY} order?",

    # C-11: Email with person, org, two commodities
    "{PERSON} ({ORG}) requested {COMMODITY#1} and {COMMODITY#2} be dispatched by {decoy:date}.",

    # -------------------------------------------------------------------------
    # D. WEBHOOK / API PAYLOADS — JSON-looking, escaped strings
    # (§12: webhook style, ~10% of traffic)
    # -------------------------------------------------------------------------

    # D-1: JSON-style key:value with double-quote look-alikes
    "\"buyer\": \"{ORG}\", \"commodity\": \"{COMMODITY}\", \"destination\": \"{ADDRESS}\"",

    # D-2: Webhook payload with ref and date
    "event: shipment_confirmed | ref: {decoy:ref_id} | supplier: {ORG} | item: {COMMODITY} | date: {decoy:date}",

    # D-3: JSON-looking with person field
    "\"contact\": \"{PERSON}\", \"org\": \"{ORG}\", \"item\": \"{COMMODITY}\"",

    # D-4: API payload with incoterm
    "payload={{\"seller\": \"{ORG}\", \"goods\": \"{COMMODITY}\", \"incoterm\": \"{decoy:incoterm}\", \"ship_to\": \"{ADDRESS}\"}}",

    # D-5: Webhook with two commodities
    "order_id={decoy:invoice_id} items=[{COMMODITY#1},{COMMODITY#2}] buyer={ORG}",

    # D-6: API log line with date and person
    "[{decoy:date}] user={PERSON} org={ORG} action=approved commodity={COMMODITY}",

    # D-7: JSON field fragment with currency
    "\"amount\": \"{decoy:currency}{decoy:qty}\", \"commodity\": \"{COMMODITY}\", \"shipper\": \"{ORG}\"",

    # D-8: Webhook with address
    "webhook trigger: delivery_confirmed | to: {ADDRESS} | goods: {COMMODITY} | carrier: {ORG}",

    # -------------------------------------------------------------------------
    # E. OCR LINES — pipe/column drift, messy spacing, all-caps
    # (§12: OCR style, ~10% of traffic)
    # -------------------------------------------------------------------------

    # E-1: Pipe-delimited OCR'd line with drift
    "{ORG}|{COMMODITY}|{decoy:qty}|{ADDRESS}",

    # E-2: OCR label:value with person
    "CONSIGNEE:{ORG}  CONTACT:{PERSON}  ITEM:{COMMODITY}",

    # E-3: OCR'd invoice header with tabs
    "{decoy:invoice_id}\t{ORG}\t{COMMODITY}\t{decoy:currency}{decoy:qty}",

    # E-4: OCR column-drift line
    "{PERSON}      {ORG}      {ADDRESS}",

    # E-5: OCR with no separators (adjacency case §6.4)
    "{PERSON}{ORG}",

    # E-6: OCR pipe-delimited tight
    "{PERSON}|{ORG}|{COMMODITY}",

    # E-7: OCR'd manifest line with commodity and address
    "CARGO: {COMMODITY}  DEST: {ADDRESS}  CARRIER: {ORG}",

    # E-8: OCR with ref id and commodity
    "BL NO {decoy:ref_id} COMMODITY {COMMODITY} SHIPPER {ORG}",

    # -------------------------------------------------------------------------
    # F. 0-ENTITY RECORDS — zero-entity true negatives (§10, §6.1)
    # -------------------------------------------------------------------------

    # F-1: Pure boilerplate
    "{decoy:boilerplate}",

    # F-2: Date + boilerplate
    "{decoy:date} — {decoy:boilerplate}",

    # F-3: Signoff only
    "{decoy:signoff}",

    # F-4: Ref id + date
    "{decoy:ref_id} — {decoy:date}",

    # F-5: Invoice id + boilerplate
    "{decoy:invoice_id}: {decoy:boilerplate}",

    # F-6: Salutation + boilerplate
    "{decoy:salutation} {decoy:boilerplate}",

    # F-7: Signoff + date
    "{decoy:signoff} {decoy:date}",

    # F-8: Two boilerplate sentences
    "{decoy:boilerplate} {decoy:boilerplate}",

    # F-9: Ref, date, boilerplate — looks data-like but no entities
    "Ref: {decoy:ref_id} | Date: {decoy:date} | {decoy:boilerplate}",

    # -------------------------------------------------------------------------
    # G. 1-ENTITY RECORDS (§6.1: single entity, ~20% target)
    # -------------------------------------------------------------------------

    # G-1: Single commodity
    "Invoice: {decoy:qty} {COMMODITY}.",

    # G-2: Single org
    "{ORG} acknowledges receipt.",

    # G-3: Single person
    "{decoy:salutation} {PERSON},",

    # G-4: Single address
    "Deliver to: {ADDRESS}.",

    # G-5: Single commodity with packaging
    "{decoy:packaging} {COMMODITY} approved for export.",

    # G-6: Single person with role
    "{decoy:role} {PERSON} signed the release.",

    # G-7: Single org with incoterm
    "{ORG} ships under {decoy:incoterm} terms.",

    # G-8: Single commodity with currency
    "{COMMODITY} valued at {decoy:currency}{decoy:qty}.",

    # G-9: Single person with title and signoff
    "{decoy:signoff} {decoy:title} {PERSON}",

    # -------------------------------------------------------------------------
    # H. 2-ENTITY RECORDS — TYPE CO-OCCURRENCE PAIRS (§6.2)
    # -------------------------------------------------------------------------

    # H-1: PERSON + ORG
    "{decoy:role} {PERSON} works at {ORG}.",

    # H-2: PERSON + ADDRESS
    "Please dispatch to {ADDRESS} c/o {PERSON}.",

    # H-3: PERSON + COMMODITY
    "{PERSON} approved the release of {COMMODITY}.",

    # H-4: ORG + ADDRESS
    "{ORG} is located at {ADDRESS}.",

    # H-5: ORG + COMMODITY
    "{ORG} supplies {decoy:qty} {COMMODITY}.",

    # H-6: ADDRESS + COMMODITY
    "Ship {COMMODITY} to {ADDRESS}.",

    # H-7: COMMODITY + COMMODITY (serial)
    "{COMMODITY#1} and {COMMODITY#2} cleared customs.",

    # H-8: PERSON + ORG variant
    "{PERSON} ({ORG}) confirmed the order.",

    # H-9: ORG + COMMODITY with incoterm
    "{ORG} delivers {COMMODITY} {decoy:incoterm}.",

    # H-10: ORG + ADDRESS (ship-to)
    "Consignee: {ORG}, delivery address: {ADDRESS}.",

    # H-11: PERSON + COMMODITY with qty
    "{PERSON} requested {decoy:qty} {COMMODITY}.",

    # H-12: ORG + ORG (multi-party)
    "{ORG#1} transferred goods to {ORG#2}.",

    # -------------------------------------------------------------------------
    # I. 3-ENTITY RECORDS (§6.1: 3 entities, ~25% target)
    # -------------------------------------------------------------------------

    # I-1: PERSON + ORG + COMMODITY
    "{PERSON} at {ORG} confirmed shipment of {COMMODITY}.",

    # I-2: ORG + COMMODITY + ADDRESS
    "{ORG} ships {COMMODITY} to {ADDRESS}.",

    # I-3: PERSON + ORG + ADDRESS
    "{PERSON} from {ORG} operates out of {ADDRESS}.",

    # I-4: PERSON + COMMODITY + ADDRESS
    "{decoy:title} {PERSON} arranged {COMMODITY} delivery to {ADDRESS}.",

    # I-5: ORG + COMMODITY#1 + COMMODITY#2
    "{ORG} certifies: {COMMODITY#1} and {COMMODITY#2} passed inspection.",

    # I-6: PERSON + ORG + COMMODITY with role
    "{decoy:role} {PERSON} of {ORG} signed for {decoy:qty} {COMMODITY}.",

    # I-7: ORG + ADDRESS + COMMODITY with ref
    "Per {decoy:ref_id}: {ORG} at {ADDRESS} to supply {COMMODITY}.",

    # I-8: Two persons + one commodity
    "{PERSON#1} and {PERSON#2} jointly approved {COMMODITY} release.",

    # I-9: ORG + two commodities with currency
    "{ORG}: {decoy:currency}{decoy:qty} for {COMMODITY#1} and {decoy:currency}{decoy:qty} for {COMMODITY#2}.",

    # -------------------------------------------------------------------------
    # J. 4-ENTITY RECORDS — full invoice (§6.1: 4 entities, ~15% target)
    # -------------------------------------------------------------------------

    # J-1: Canonical four-type invoice
    "{decoy:title} {PERSON} from {ORG} confirmed {COMMODITY} shipped to {ADDRESS}.",

    # J-2: With ref and incoterm
    "Ref {decoy:ref_id}: {ORG} confirms {COMMODITY} {decoy:incoterm} to {ADDRESS}, attention {PERSON}.",

    # J-3: With qty and date
    "On {decoy:date}, {PERSON} ({ORG}) arranged {decoy:qty} {COMMODITY} for delivery to {ADDRESS}.",

    # J-4: Email style four-type
    "{decoy:salutation} {PERSON}, per {decoy:invoice_id}, {ORG} will ship {COMMODITY} to {ADDRESS}.",

    # J-5: Tabular four-type
    "Buyer: {PERSON} | Org: {ORG} | Item: {COMMODITY} | Ship-to: {ADDRESS}",

    # J-6: Four-type with currency
    "{ORG} invoices {PERSON} {decoy:currency}{decoy:qty} for {COMMODITY} to {ADDRESS}.",

    # J-7: Manifest-style four-type
    "Manifest {decoy:ref_id}: {PERSON} at {ORG} ships {COMMODITY} to {ADDRESS}.",

    # -------------------------------------------------------------------------
    # K. 5+ ENTITY RECORDS — long manifests, serial lists (§6.1: 5+ entities)
    # -------------------------------------------------------------------------

    # K-1: Three commodities + person + org (5 entities)
    "{PERSON} ({ORG}) ships {COMMODITY#1}, {COMMODITY#2}, and {COMMODITY#3} per {decoy:invoice_id}.",

    # K-2: Four commodities + org (5 entities)
    "{ORG} declares cargo: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}, {COMMODITY#4}.",

    # K-3: Five commodities + org (6 entities)
    "Manifest: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3}, {COMMODITY#4}, {COMMODITY#5} from {ORG}.",

    # K-4: Two persons + two orgs + one commodity (5 entities)
    "{PERSON#1} ({ORG#1}) and {PERSON#2} ({ORG#2}) jointly approved {COMMODITY}.",

    # K-5: Person + org + three commodities + address (6 entities)
    "{PERSON} at {ORG} loaded {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} for dispatch to {ADDRESS}.",

    # K-6: Long manifest with five commodities (5 entities)
    "Cargo manifest {decoy:ref_id}: {decoy:qty} {COMMODITY#1}, {decoy:qty} {COMMODITY#2}, {decoy:qty} {COMMODITY#3}, {decoy:qty} {COMMODITY#4}, {decoy:qty} {COMMODITY#5} — consignee {ORG}.",

    # -------------------------------------------------------------------------
    # L. PURE DENIAL — negation only (§5.1: NEG-only bucket)
    # -------------------------------------------------------------------------

    # L-1: ORG + pure denial
    "{ORG} {decoy:neg_cue} {NEG_COMMODITY}.",

    # L-2: Invoice id + denial
    "{decoy:invoice_id}: {decoy:neg_cue} {NEG_COMMODITY}.",

    # L-3: Person-signed denial
    "{PERSON} states: {decoy:neg_cue} {NEG_COMMODITY} in this shipment.",

    # L-4: Denial with ref
    "Per {decoy:ref_id}: shipment {decoy:neg_cue} {NEG_COMMODITY}.",

    # L-5: Denial with date
    "As of {decoy:date}, cargo {decoy:neg_cue} {NEG_COMMODITY}.",

    # L-6: ORG denial with incoterm context
    "{ORG} confirms this manifest {decoy:neg_cue} {NEG_COMMODITY}.",

    # -------------------------------------------------------------------------
    # M. SCOPE DISTANCE VARIATION (§5.3)
    # -------------------------------------------------------------------------

    # M-1: Scope distance 0 — immediate (no filler tokens)
    "{decoy:neg_cue} {NEG_COMMODITY}.",

    # M-2: Scope distance 1 — one filler token after cue
    "{decoy:neg_cue} any {NEG_COMMODITY}.",

    # M-3: Scope distance 2 — two fillers
    "The shipment {decoy:neg_cue} certified {NEG_COMMODITY}.",

    # M-4: Scope distance 3 — "does not contain any trace of X"
    "{decoy:neg_cue} any trace of {NEG_COMMODITY}.",

    # M-5: Scope distance 5-10 — long intervening phrase
    "The consignment {decoy:neg_cue}, per the inspection report, contain {NEG_COMMODITY}.",

    # M-6: Scope distance 5-10 with org
    "{ORG} declares that the cargo {decoy:neg_cue}, as inspected on {decoy:date}, include {NEG_COMMODITY}.",

    # M-7: Scope distance 5-10 with person and org
    "{decoy:title} {PERSON} of {ORG} certifies this lot {decoy:neg_cue}, per customs, contain {NEG_COMMODITY}.",

    # -------------------------------------------------------------------------
    # N. WORD-ORDER A — denial then assertion (§5: word-order A)
    # -------------------------------------------------------------------------

    # N-1: Classic denial-then-assertion
    "{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}.",

    # N-2: With person
    "Shipment {decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}, per {PERSON}.",

    # N-3: With org
    "{ORG} confirms: {decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}.",

    # N-4: With qty on asserted side
    "{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {decoy:qty} {COMMODITY#2}.",

    # N-5: Tabular denial-then-assertion
    "DENY: {decoy:neg_cue} {NEG_COMMODITY#1} | ACCEPT: {COMMODITY#2}",

    # N-6: With address
    "{decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2} for delivery to {ADDRESS}.",

    # -------------------------------------------------------------------------
    # O. WORD-ORDER B — assertion then denial (§5: word-order B)
    # -------------------------------------------------------------------------

    # O-1: Classic assertion-then-denial
    "{COMMODITY#1}, but {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # O-2: With org
    "Delivery contains {COMMODITY#1}, but {decoy:neg_cue} {NEG_COMMODITY#2} ({ORG}).",

    # O-3: With person
    "{PERSON} approved {COMMODITY#1} but noted {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # O-4: With qty on asserted side
    "{decoy:qty} {COMMODITY#1} confirmed; {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # O-5: Multi-word assertion then denial
    "Manifest lists {COMMODITY#1} and {COMMODITY#2}, but {decoy:neg_cue} {NEG_COMMODITY#3}.",

    # O-6: Tabular assertion-then-denial
    "{ORG}: {COMMODITY#1} approved | {decoy:neg_cue} {NEG_COMMODITY#2}",

    # -------------------------------------------------------------------------
    # P. MULTI-TARGET NEGATION (§5: multi-target NEG, one cue + 2-3 denied)
    # -------------------------------------------------------------------------

    # P-1: Three denied commodities
    "{decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}, or {NEG_COMMODITY#3}.",

    # P-2: Two denied commodities with org
    "{ORG} certifies: {decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}.",

    # P-3: Two denied with ref
    "Per {decoy:ref_id}: {decoy:neg_cue} {NEG_COMMODITY#1} or {NEG_COMMODITY#2}.",

    # P-4: Three denied with person
    "{PERSON} confirmed this lot {decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}, or {NEG_COMMODITY#3}.",

    # P-5: Two denied with address
    "Goods from {ADDRESS} {decoy:neg_cue} {NEG_COMMODITY#1} or {NEG_COMMODITY#2}.",

    # P-6: Three denied with org and person
    "{ORG} and {PERSON} jointly certify: {decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}, {NEG_COMMODITY#3}.",

    # -------------------------------------------------------------------------
    # Q. MIXED POS + NEG IN SAME RECORD (§5.1)
    # -------------------------------------------------------------------------

    # Q-1: Qualified NEG + bare POS
    "{decoy:neg_cue} {NEG_COMMODITY#1}; ordinary {COMMODITY#2} acceptable.",

    # Q-2: ORG + POS + NEG
    "{ORG} requires {COMMODITY#1}, {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # Q-3: Four entities + mixed polarity
    "{PERSON} at {ORG}: deliver {COMMODITY#1} to {ADDRESS}, {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # Q-4: Two POS + one NEG
    "{COMMODITY#1} and {COMMODITY#2} approved; {decoy:neg_cue} {NEG_COMMODITY#3}.",

    # Q-5: Denial-assertion with full context
    "Per {decoy:invoice_id}: {decoy:neg_cue} {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2}, ship to {ADDRESS}.",

    # Q-6: Mixed with person and signoff
    "{decoy:salutation} {PERSON}, please include {COMMODITY#1} but {decoy:neg_cue} {NEG_COMMODITY#2}. {decoy:signoff}",

    # Q-7: Three-commodity mixed (two POS + one NEG)
    "Approved: {COMMODITY#1}, {COMMODITY#2}. {decoy:neg_cue} {NEG_COMMODITY#3}.",

    # -------------------------------------------------------------------------
    # R. FROZEN COMPOUNDS — dissolved negation cues, commodity stays POS (§5.2)
    # -------------------------------------------------------------------------

    # R-1: ORG + frozen compound + commodity
    "{ORG} supplies {decoy:frozen_compound} {COMMODITY}.",

    # R-2: Person + frozen compound + commodity
    "{PERSON} confirmed {decoy:frozen_compound} {COMMODITY} delivery.",

    # R-3: Qty + frozen compound + commodity + address
    "{decoy:qty} {decoy:frozen_compound} {COMMODITY} shipped to {ADDRESS}.",

    # R-4: Invoice with frozen compound
    "{decoy:invoice_id}: {decoy:qty} {decoy:frozen_compound} {COMMODITY} for {ORG}.",

    # R-5: Email with frozen compound
    "{decoy:salutation} {PERSON}, please confirm {decoy:frozen_compound} {COMMODITY} from {ORG}.",

    # R-6: Manifest with frozen compound
    "Manifest: {decoy:frozen_compound} {COMMODITY#1} and {COMMODITY#2} loaded at {ADDRESS}.",

    # R-7: Frozen compound with positive assertion (no NEG_COMMODITY in record)
    "Order {decoy:invoice_id}: {decoy:frozen_compound} {COMMODITY}, {decoy:qty}, shipped by {ORG} to {ADDRESS}.",

    # R-8: Two frozen compound commodities
    "{ORG} confirmed {decoy:frozen_compound} {COMMODITY#1} and {decoy:frozen_compound} {COMMODITY#2}.",

    # R-9: Frozen compound + mixed denial in same record (frozen stays POS)
    "{PERSON} confirmed {decoy:frozen_compound} {COMMODITY#1} acceptable; {decoy:neg_cue} {NEG_COMMODITY#2}.",

    # -------------------------------------------------------------------------
    # S. BOUNDARY / QUANTITY — qty/packaging BEFORE commodity, title/role BEFORE person
    # (§8.1: quantity excluded from span; §8.3: title excluded from span)
    # -------------------------------------------------------------------------

    # S-1: qty before commodity
    "Invoice: {decoy:qty} {COMMODITY} for {ORG}.",

    # S-2: packaging before commodity
    "{decoy:packaging} {COMMODITY} cleared at {ADDRESS}.",

    # S-3: qty + packaging + commodity
    "{decoy:qty} {decoy:packaging} {COMMODITY} approved by {PERSON}.",

    # S-4: title + person
    "{decoy:title} {PERSON} signed the manifest.",

    # S-5: role + person
    "{decoy:role} {PERSON} from {ORG} approved {decoy:qty} {COMMODITY}.",

    # S-6: salutation before person
    "{decoy:salutation} {PERSON} of {ORG}, the {COMMODITY} is ready.",

    # S-7: unit + commodity
    "{decoy:unit} of {COMMODITY} per {decoy:ref_id}.",

    # S-8: qty + commodity + unit
    "{ORG} ships {decoy:qty} {COMMODITY} ({decoy:unit}).",

    # -------------------------------------------------------------------------
    # T. REPEATED ENTITIES & ADJACENCY (§6.3, §6.4)
    # -------------------------------------------------------------------------

    # T-1: Same org twice (repeated entity §6.3)
    "{ORG#1} | {ORG#1}",

    # T-2: No-separator adjacency PERSON + ORG (§6.4)
    "{PERSON}{ORG}",

    # T-3: Tight pipe separator
    "{PERSON}|{ORG}",

    # T-4: Two consecutive persons no separator (§9: adversarial)
    "{PERSON#1} {PERSON#2}",

    # T-5: Comma-separated adjacency
    "{PERSON}, {ORG}.",

    # T-6: Punctuation-only separator
    "{PERSON}. {ORG}.",

    # T-7: Same commodity referenced twice in one record
    "Supplier has {COMMODITY#1}; buyer requests {COMMODITY#1}.",

    # T-8: Two commodities same head noun (red wine / white wine style)
    "{COMMODITY#1} and {COMMODITY#2} are both approved grades.",

    # T-9: Org with alias pattern
    "{ORG#1} ({ORG#1}) acknowledges receipt of {COMMODITY}.",

    # T-10: PERSON tight adjacent to COMMODITY
    "{PERSON}:{COMMODITY}",

    # -------------------------------------------------------------------------
    # U. CURRENCY, INCOTERM, DATE EXTRAS (§12 invoice features)
    # -------------------------------------------------------------------------

    # U-1: Currency in invoice line
    "{decoy:currency}{decoy:qty} for {COMMODITY} from {ORG}.",

    # U-2: Incoterm + address + commodity
    "{COMMODITY} delivered {decoy:incoterm} {ADDRESS} by {ORG}.",

    # U-3: Date + ref + person + commodity
    "{decoy:date} | {decoy:ref_id} | {PERSON} | {COMMODITY}",

    # U-4: Currency + org + person
    "{decoy:currency}{decoy:qty} paid to {ORG} per {decoy:title} {PERSON}.",

    # U-5: Incoterm + three commodities
    "{decoy:incoterm}: {COMMODITY#1}, {COMMODITY#2}, {COMMODITY#3} from {ORG}.",

    # -------------------------------------------------------------------------
    # V. EMAIL SALUTATION + PERSON + SIGNOFF shape (§12 email style)
    # -------------------------------------------------------------------------

    # V-1: Salutation + person name at top, signoff at bottom
    "{decoy:salutation} {PERSON},\n\nPlease find attached the invoice for {COMMODITY} from {ORG}.\n\n{decoy:signoff}",

    # V-2: Salutation + two persons
    "{decoy:salutation} {PERSON#1} and {PERSON#2}, the {COMMODITY} shipment is confirmed.",

    # V-3: Body with signoff person
    "We confirm delivery of {decoy:qty} {COMMODITY} to {ADDRESS}.\n\n{decoy:signoff} {PERSON}",

    # V-4: Full email with all four types
    "{decoy:salutation} {PERSON},\n\n{ORG} will deliver {COMMODITY} to {ADDRESS} on {decoy:date}.\n\n{decoy:signoff}",

    # -------------------------------------------------------------------------
    # W. ADDITIONAL MIXED / EDGE PATTERNS to reach 130–170 total
    # -------------------------------------------------------------------------

    # W-1: Webhook/JSON with neg commodity
    "\"status\": \"denied\", \"commodity\": \"{NEG_COMMODITY}\", \"reason\": \"{decoy:neg_cue}\"",

    # W-2: OCR line with qty + neg cue
    "CARGO {decoy:neg_cue} {NEG_COMMODITY} PER {decoy:ref_id}",

    # W-3: Manifest multi-commodity with one NEG
    "Manifest {decoy:ref_id}: {COMMODITY#1}, {COMMODITY#2} cleared; {decoy:neg_cue} {NEG_COMMODITY#3}.",

    # W-4: Invoice with person, org, address, two pos commodities (5 entities)
    "{decoy:invoice_id}: {PERSON} ({ORG}) — {COMMODITY#1}, {COMMODITY#2} shipped to {ADDRESS}.",

    # W-5: ORG + two addresses
    "{ORG} ships from {ADDRESS#1} to {ADDRESS#2}.",

    # W-6: Tab-delimited denial
    "{decoy:neg_cue}\t{NEG_COMMODITY}\t{ORG}\t{decoy:date}",

    # W-7: Multi-line email with neg and pos
    "{decoy:salutation} {PERSON},\n\nApproved: {COMMODITY#1}.\n{decoy:neg_cue} {NEG_COMMODITY#2}.\n\n{decoy:signoff}",

    # W-8: Frozen compound in negation-heavy context (stays POS)
    "While {decoy:neg_cue} {NEG_COMMODITY#1}, we do stock {decoy:frozen_compound} {COMMODITY#2} at {ADDRESS}.",

    # W-9: Mixed polarity 4-entity invoice
    "{PERSON} at {ORG}: {COMMODITY#1} approved, {decoy:neg_cue} {NEG_COMMODITY#2}, deliver to {ADDRESS}.",

    # W-10: Two-address, two-commodity manifest (4 entities)
    "Ship {COMMODITY#1} to {ADDRESS#1}; also {COMMODITY#2} to {ADDRESS#2}, care of {ORG}.",

    # W-11: Webhook with person + neg commodity
    "alert: {decoy:neg_cue} {NEG_COMMODITY} | contact: {PERSON} | org: {ORG}",

    # W-12: Long scope distance (5-10 tokens) with org
    "{ORG}, in accordance with the latest regulatory guidance, {decoy:neg_cue} {NEG_COMMODITY}.",

    # W-13: Scope distance 2-4 with contrast cue
    "{decoy:neg_cue} any {NEG_COMMODITY#1}{decoy:contrast_cue} {COMMODITY#2} to {ADDRESS}.",

    # W-14: Signoff + person (no other entities)
    "{decoy:signoff} {PERSON}",

    # W-15: Boilerplate + ref (zero-entity variant)
    "{decoy:boilerplate} Reference: {decoy:ref_id}.",

    # W-16: Four-type with packaging and incoterm
    "{PERSON} ({ORG}): {decoy:packaging} {COMMODITY}, {decoy:incoterm}, to {ADDRESS}.",

    # W-17: Serial commodities with qty + org (5 entities)
    "{ORG} ships {decoy:qty} {COMMODITY#1}, {decoy:qty} {COMMODITY#2}, {decoy:qty} {COMMODITY#3}, {decoy:qty} {COMMODITY#4}.",

    # W-18: OCR-style with person adjacent to commodity (tight)
    "{PERSON}|{COMMODITY}|{ADDRESS}",

    # W-19: Denial with incoterm context
    "Under {decoy:incoterm} terms, {ORG} confirms this shipment {decoy:neg_cue} {NEG_COMMODITY}.",

    # W-20: Multi-target denial + positive in same record (5 entities)
    "{decoy:neg_cue} {NEG_COMMODITY#1}, {NEG_COMMODITY#2}; acceptable goods: {COMMODITY#3} from {ORG}.",

]
