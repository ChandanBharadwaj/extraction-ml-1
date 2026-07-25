"""COMMODITY entity pool for the synthetic NER training-data seed.

Contract references (docs/data_specification.md):
  §4.4  COMMODITY surface forms and boundary rules
  §5.4  Subtype-qualifier discipline: NEG must cover the full qualified phrase,
        so bare + qualified variants of the same head noun must both appear.
  §8.1  Boundary discipline: exclude quantity / unit / packaging words;
        include grade, treatment, variety, alphanumeric product codes.
  §3.x  Length buckets: XS (1–3 chars), S (4–10), M (11–25), L (26–60),
        XL (61–120), XXL (121+). All buckets represented.
  §11.3 Industry diversity: Energy, Metals, Agricultural, Chemical/plastics,
        Consumer goods, Pharmaceuticals — each vertical ≥ 30 distinct values.

Every value here is the EXACT gold span text.  No quantity, unit, or
packaging prefix is included.  Plain Python strings; the downstream builder
handles SQL escaping (values with apostrophes use double-quote delimiters).
"""

COMMODITIES: list[str] = [

    # -------------------------------------------------------------------------
    # ENERGY  (§11.3 — ≥30 values)
    # bare + qualified pairs: oil, crude, LNG, coal, gas, naphtha, diesel …
    # -------------------------------------------------------------------------

    # XS
    "oil",
    "gas",
    "LNG",
    "LPG",

    # bare head nouns
    "crude",
    "naphtha",
    "diesel",
    "gasoline",
    "kerosene",
    "bitumen",
    "asphalt",

    # qualified forms (treatment / grade / variety)
    "Brent crude",
    "WTI crude",
    "Dubai crude",
    "OPEC crude",
    "light sweet crude",
    "heavy sour crude",
    "crude oil",
    "Brent crude oil",
    "WTI crude oil",
    "condensate",
    "natural gas condensate",
    "liquefied natural gas",
    "liquefied petroleum gas",
    "pipeline natural gas",
    "jet fuel",
    "Jet A-1 fuel",
    "aviation turbine fuel",
    "ultra-low sulphur diesel",
    "high-speed diesel",
    "marine fuel oil",
    "heavy fuel oil",
    "residual fuel oil",
    "fuel oil",
    "refined petroleum products",
    "thermal coal",
    "coking coal",
    "metallurgical coal",
    "steam coal",
    "coal",
    "petroleum coke",
    "petcoke",
    "naphtha feedstock",
    "reformate",
    "vacuum gas oil",
    "straight-run naphtha",
    "mixed xylenes",

    # -------------------------------------------------------------------------
    # METALS  (§11.3 — ≥30 values)
    # bare + qualified pairs: steel, copper, aluminum, zinc, lead, nickel …
    # -------------------------------------------------------------------------

    # XS
    "tin",
    "ore",
    "zinc",
    "lead",

    # bare head nouns
    "steel",
    "copper",
    "aluminum",
    "aluminium",
    "nickel",
    "iron",
    "silver",
    "gold",
    "platinum",
    "palladium",
    "cobalt",
    "titanium",
    "manganese",
    "chromium",
    "molybdenum",
    "tungsten",
    "vanadium",

    # qualified / treatment / grade forms
    "steel sheet",
    "stainless steel",
    "stainless steel sheet",
    "304 stainless steel sheet",
    "316L stainless steel sheet",
    "galvanized steel coil",
    "hot-rolled coil",
    "cold-rolled coil",
    "hot-rolled steel coil",
    "cold-rolled steel coil",
    "steel rebar",
    "deformed steel bar",
    "steel wire rod",
    "steel billet",
    "steel slab",
    "steel pipe",
    "seamless steel pipe",
    "structural steel",
    "high-tensile steel",
    "carbon steel plate",
    "alloy steel bar",
    "iron ore",
    "iron ore fines",
    "iron ore pellets",
    "sintered iron ore",
    "copper cathode",
    "refined copper cathode",
    "LME copper",
    "copper rod",
    "copper wire",
    "copper concentrate",
    "aluminum ingot",
    "primary aluminum ingot",
    "aluminum billet",
    "aluminum coil",
    "aluminum alloy ingot",
    "zinc ingot",
    "refined zinc",
    "lead ingot",
    "refined lead",
    "nickel cathode",
    "electrolytic nickel",
    "nickel briquette",
    "tin ingot",
    "refined tin",
    "gold bullion",
    "gold bar",
    "London Good Delivery gold bar",
    "silver bullion",
    "silver bar",
    "platinum sponge",
    "palladium sponge",

    # -------------------------------------------------------------------------
    # AGRICULTURAL  (§11.3 — ≥30 values)
    # bare + qualified pairs: coffee, sugar, cotton, wheat, rice, palm oil …
    # -------------------------------------------------------------------------

    # XS
    "rice",

    # bare head nouns
    "coffee",
    "sugar",
    "cotton",
    "wheat",
    "corn",
    "maize",
    "soybean",
    "rubber",
    "cocoa",
    "tea",
    "tobacco",
    "barley",
    "oats",
    "rapeseed",
    "sunflower",
    "jute",

    # qualified forms
    "robusta coffee",
    "arabica coffee",
    "Grade A robusta coffee",
    "washed arabica coffee",
    "natural arabica coffee",
    "green coffee beans",
    "roasted coffee beans",
    "raw cane sugar",
    "cane sugar",
    "white refined sugar",
    "white sugar",
    "brown sugar",
    "molasses",
    "organic cotton",
    "raw cotton",
    "ginned cotton",
    "combed cotton yarn",
    "cotton lint",
    "hard red winter wheat",
    "soft red winter wheat",
    "white wheat",
    "durum wheat",
    "wheat flour",
    "yellow corn",
    "white corn",
    "corn starch",
    "soybean meal",
    "soybean oil",
    "crude soybean oil",
    "refined soybean oil",
    "palm oil",
    "crude palm oil",
    "refined palm oil",
    "palm olein",
    "palm kernel oil",
    "crude palm kernel oil",
    "basmati rice",
    "long-grain white rice",
    "parboiled rice",
    "milled rice",
    "cocoa beans",
    "fermented cocoa beans",
    "cocoa butter",
    "cocoa powder",
    "natural rubber",
    "ribbed smoked sheet rubber",
    "technically specified rubber",
    "black tea",
    "green tea",
    "oolong tea",
    "flue-cured tobacco",

    # -------------------------------------------------------------------------
    # CHEMICAL / PLASTICS  (§11.3 — ≥30 values)
    # bare + qualified pairs: ammonia, urea, ethylene, HDPE, LDPE, PVC …
    # -------------------------------------------------------------------------

    # bare head nouns
    "ammonia",
    "urea",
    "methanol",
    "ethanol",
    "ethylene",
    "propylene",
    "benzene",
    "toluene",
    "xylene",
    "styrene",
    "acetone",
    "chlorine",
    "caustic soda",
    "soda ash",
    "sulphuric acid",
    "hydrochloric acid",
    "nitric acid",
    "phosphoric acid",
    "sodium hydroxide",

    # qualified / treatment / grade / alphanumeric product codes
    "anhydrous ammonia",
    "liquid ammonia",
    "granular urea",
    "prilled urea",
    "urea 46%",
    "monoammonium phosphate",
    "diammonium phosphate",
    "NPK fertiliser",
    "potassium chloride",
    "muriate of potash",
    "HDPE",
    "LDPE",
    "LLDPE",
    "polypropylene",
    "PVC resin",
    "polystyrene",
    "ABS resin",
    "PET resin",
    "nylon 6",
    "nylon 6,6",
    "polyurethane",
    "epoxy resin",
    "phenol formaldehyde resin",
    "Polyethylene resin HDPE",
    "HDPE 5502",
    "HDPE pipe grade resin",
    "LDPE 1810D",
    "LLDPE film grade",
    "polypropylene homopolymer",
    "polypropylene copolymer",
    "PVC suspension resin",
    "PVC paste resin",
    "expandable polystyrene",
    "high-impact polystyrene",

    # XL / XXL — long alphanumeric product codes (§3.2)
    "high-density polyethylene injection-moulding grade resin HDPE 5502 natural",
    "low-density polyethylene tubular process film grade LDPE 1922T pellets",
    "linear low-density polyethylene C6 hexene copolymer film grade LLDPE 218W",
    "polypropylene random copolymer pipe grade PP-R 4220 natural pellets",
    "polyvinyl chloride suspension resin general-purpose grade PVC S-65 powder",

    # -------------------------------------------------------------------------
    # CONSUMER GOODS  (§11.3 — ≥30 values)
    # electronics, apparel, packaged foods, household goods …
    # -------------------------------------------------------------------------

    # bare head nouns
    "electronics",
    "smartphones",
    "laptops",
    "tablets",
    "apparel",
    "footwear",
    "furniture",
    "toys",
    "cosmetics",
    "beverages",
    "chocolate",
    "flour",

    # qualified / variety / frozen-compound positives (§4.4, §5.1)
    "LED panels",
    "LED television panels",
    "LCD display modules",
    "OLED screens",
    "lithium-ion battery cells",
    "lithium-ion battery packs",
    "consumer electronics",
    "mobile handsets",
    "feature phones",
    "wireless earbuds",
    "smart home devices",
    "cotton T-shirts",
    "knitted cotton shirts",
    "denim jeans",
    "polyester sportswear",
    "woven synthetic fabric",
    "leather shoes",
    "athletic footwear",
    "rubber-soled footwear",
    "flat-pack furniture",
    "upholstered sofas",
    "packaged foods",
    "canned tuna",
    "canned sardines",
    "instant noodles",
    "breakfast cereals",
    "biscuits and cookies",
    "bottled water",
    "carbonated soft drinks",
    "fruit juice concentrate",
    "dark chocolate",
    "milk chocolate",
    "sugar-free chocolate",       # frozen compound: POS (§4.4, §5.2)
    "gluten-free flour",          # frozen compound: POS
    "non-stick pan",              # frozen compound: POS
    "lead-free solder",           # frozen compound: POS
    "stainless steel cookware",   # frozen compound embedded: POS
    "dairy-free milk alternative",
    "personal care products",
    "baby diapers",
    "laundry detergent",
    "dish soap",
    "shampoo",

    # -------------------------------------------------------------------------
    # PHARMACEUTICALS  (§11.3 — ≥30 values)
    # APIs, finished dosage, biologics …
    # -------------------------------------------------------------------------

    # bare head nouns
    "paracetamol",
    "ibuprofen",
    "amoxicillin",
    "aspirin",
    "metformin",
    "insulin",
    "penicillin",
    "azithromycin",
    "atorvastatin",
    "omeprazole",
    "ciprofloxacin",
    "dexamethasone",
    "hydroxychloroquine",
    "oseltamivir",
    "remdesivir",

    # qualified forms (API, finished dosage, grade)
    "paracetamol API",
    "ibuprofen API",
    "amoxicillin trihydrate",
    "amoxicillin capsules",
    "active pharmaceutical ingredient",
    "finished dosage tablets",
    "finished dosage forms",
    "film-coated tablets",
    "paracetamol 500 mg tablets",    # dose is inseparable product identity here
    "ibuprofen 400 mg film-coated tablets",
    "metformin hydrochloride tablets",
    "atorvastatin calcium tablets",
    "omeprazole delayed-release capsules",
    "azithromycin dihydrate powder",
    "ciprofloxacin hydrochloride tablets",
    "dexamethasone sodium phosphate injection",
    "insulin vials",
    "insulin glargine injection",
    "human insulin 100 IU/mL",
    "monoclonal antibody",
    "recombinant human albumin",
    "lyophilised vaccine bulk",
    "intravenous immunoglobulin",
    "heparin sodium injection",
    "morphine sulphate tablets",
    "amoxicillin-clavulanate potassium tablets",
    "co-amoxiclav 625 mg tablets",
    "oral rehydration salts",
    "vitamin C ascorbic acid USP grade",
    "folic acid USP grade",

    # -------------------------------------------------------------------------
    # §5.4 BARE + QUALIFIED HEAD-NOUN FAMILIES
    # (many already appear above; this section adds the bare forms that complete
    # the pairing so templates can combine e.g. "no special wood, but wood is OK")
    # -------------------------------------------------------------------------

    # wood family
    "wood",
    "special wood",
    "treated wood",
    "ordinary wood",
    "pressure-treated wood",
    "kiln-dried wood",
    "hardwood",
    "softwood",
    "engineered wood",
    "plywood",
    "marine-grade plywood",

    # coffee (additional bare — "coffee" already listed above)
    # qualified variants already listed: robusta coffee, arabica coffee, etc.

    # steel (additional bare — "steel" already listed above)
    # qualified variants already listed: steel sheet, stainless steel sheet, etc.

    # sugar (bare "sugar" already listed above)
    # qualified: raw cane sugar, cane sugar, white sugar already listed

    # copper (bare "copper" already listed above)
    # qualified: copper cathode, refined copper cathode already listed

    # ammonia (bare "ammonia" already listed above)
    # qualified: anhydrous ammonia already listed

    # cotton (bare "cotton" already listed above)
    # qualified: organic cotton, raw cotton already listed

    # additional family: wheat (bare already listed)
    # qualified: hard red winter wheat, durum wheat etc. already listed

    # additional family: palm oil (bare "palm oil" already listed)
    # qualified variants already listed above

    # additional family: nickel (bare "nickel" already listed)
    # qualified: nickel cathode, electrolytic nickel already listed

    # additional family: ethylene
    "ethylene glycol",
    "monoethylene glycol",
    "diethylene glycol",

    # additional family: methanol (bare already listed)
    "fuel grade methanol",
    "chemical grade methanol",

    # additional family: urea (bare already listed)
    # qualified: granular urea, prilled urea, urea 46% already listed

    # additional family: rubber (bare already listed)
    # qualified: natural rubber, ribbed smoked sheet rubber already listed

    # additional family: cocoa (bare already listed)
    # qualified: cocoa beans, fermented cocoa beans, cocoa butter already listed

    # additional family: coal (bare already listed)
    # qualified: thermal coal, coking coal, metallurgical coal already listed

    # additional family: rice (bare already listed)
    # qualified: basmati rice, long-grain white rice, parboiled rice already listed

    # additional family: corn / maize (bare "corn" and "maize" already listed)
    # qualified: yellow corn, white corn already listed

    # -------------------------------------------------------------------------
    # ADDITIONAL INDUSTRY-SPECIFIC JARGON / TRADE FORMS  (§4.4)
    # -------------------------------------------------------------------------

    "LME nickel",
    "LME zinc",
    "LME aluminum",
    "LME lead",
    "LME tin",
    "COMEX gold",
    "COMEX silver",
    "NYMEX crude",

    # shipping / trade staples
    "scrap metal",
    "ferrous scrap",
    "non-ferrous scrap",
    "copper scrap",
    "aluminium scrap",
    "recycled plastic pellets",
    "recovered paper",
    "waste paper",
    "old corrugated containers",

    # beverages
    "red wine",
    "white wine",
    "sparkling wine",
    "whisky",
    "brandy",
    "beer",

    # additional textiles / fibres
    "polyester fibre",
    "viscose fibre",
    "nylon fibre",
    "acrylic fibre",
    "wool",
    "merino wool",
    "raw silk",
    "silk yarn",

    # additional building / construction materials
    "cement",
    "Portland cement",
    "ordinary Portland cement",
    "clinker",
    "gypsum",
    "glass",
    "flat glass",
    "float glass",
    "ceramic tiles",
    "porcelain tiles",
    "marble",
    "granite slabs",

    # additional chemicals / intermediates
    "acetic acid",
    "formic acid",
    "glacial acetic acid",
    "hydrogen peroxide",
    "sodium carbonate",
    "calcium carbide",
    "titanium dioxide",
    "carbon black",
    "activated carbon",
    "silica gel",
    "zeolite",

    # fertilisers (complementing those above)
    "ammonium nitrate",
    "calcium ammonium nitrate",
    "triple superphosphate",
    "single superphosphate",
    "sulphate of potash",

    # additional agri commodities
    "groundnuts",
    "groundnut oil",
    "crude groundnut oil",
    "sesame seeds",
    "sesame oil",
    "dried milk powder",
    "skim milk powder",
    "whole milk powder",
    "whey powder",
    "butter",
    "anhydrous milk fat",
    "frozen shrimp",
    "frozen prawns",
    "dried shrimp",
    "fishmeal",
    "fish oil",
    "beef",
    "frozen beef",
    "boneless beef",
    "pork",
    "frozen pork",
    "chicken",
    "frozen chicken",

    # XL / XXL long product descriptors (§3.2 length buckets)
    # XL (61–120 chars)
    "Grade A washed arabica green coffee beans — Colombia origin",
    "first-quality ribbed smoked sheet No. 3 natural rubber RSS3 Vietnam origin",
    "food-grade refined bleached deodorised palm olein RBD 24-degree iodine value",
    "pharmaceutical-grade anhydrous ethanol 99.9% purity USP/EP specification",
    "electrolytic tough-pitch copper rod 8 mm ETP grade ASTM B49 standard",
    "hot-dip galvanized cold-rolled steel coil ASTM A653 G90 coating 0.55 mm",
    "titanium dioxide rutile grade chloride process pigment TiO2 93% min",
    "caustic soda lye 50% membrane cell grade NaOH technical specification",

    # XXL (121+ chars)
    "high-density polyethylene injection-moulding grade resin HDPE 5502 natural pellets conforming to ASTM D4703 and ISO 1872-1 standards",
    "low-density polyethylene tubular process autoclave film grade LDPE 1922T natural pellets conforming to ISO 1872-1 specification",
    "linear low-density polyethylene C6 hexene copolymer film grade LLDPE 218W natural pellets conforming to ISO 1872-1",
    "food-grade refined bleached deodorised palm olein RBD CP10 iodine value 56 minimum packed in ISO flexitanks",
    "pharmaceutical-grade paracetamol active pharmaceutical ingredient micronised powder BP/USP/EP tri-compendial specification",

    # =========================================================================
    # CARGO / TRADE FORMS — bills of lading, manifests, packing lists,
    # customs declarations (§4.4 jargon, §4.8 brands, §8.1 boundary rules).
    # Every value remains the EXACT gold span: no quantity, container count,
    # or packaging phrase inside a value (those live in decoy pools).
    # =========================================================================

    # --- generic-cargo declarations: the hardest COMMODITY spans because
    # --- they are generic nouns, yet they dominate real manifests
    "general cargo",
    "general merchandise",
    "consolidated cargo",
    "groupage cargo",
    "FAK cargo",
    "freight all kinds",
    "personal effects",
    "used personal effects",
    "household goods",
    "used household goods",
    "used machinery",
    "used agricultural machinery",
    "machinery spare parts",
    "spare parts",
    "auto spare parts",
    "machinery parts NOS",
    "used vehicles",
    "used tyres",
    "scrap metal",
    "mixed scrap metal",
    "dangerous goods",
    "DANGEROUS GOODS",

    # --- HS-heading-style legalese descriptions (customs registers write
    # --- these in ALL CAPS; embedded "NOT"/"OTHER THAN" are part of the
    # --- heading text, not polarity cues — hard frozen-negation training)
    "UREA, WHETHER OR NOT IN AQUEOUS SOLUTION",
    "COFFEE, NOT ROASTED, NOT DECAFFEINATED",
    "FROZEN SHRIMPS AND PRAWNS, HEADLESS SHELL-ON, INDIVIDUALLY QUICK FROZEN",
    "PORTLAND CEMENT, WHETHER OR NOT COLOURED, OTHER THAN WHITE CEMENT",
    "SEMI-MILLED OR WHOLLY MILLED RICE, WHETHER OR NOT POLISHED OR GLAZED",
    "CANE SUGAR, RAW, IN SOLID FORM, NOT CONTAINING ADDED FLAVOURING OR COLOURING MATTER",
    "FLAT-ROLLED PRODUCTS OF IRON OR NON-ALLOY STEEL, HOT-ROLLED, NOT CLAD, PLATED OR COATED",
    "POLYMERS OF ETHYLENE, IN PRIMARY FORMS, HAVING A SPECIFIC GRAVITY OF 0.94 OR MORE",
    "NEW PNEUMATIC TYRES, OF RUBBER, OF A KIND USED ON MOTOR CARS",
    "PARTS AND ACCESSORIES OF THE MOTOR VEHICLES OF HEADINGS 8701 TO 8705",
    "FROZEN FILLETS OF ALASKA POLLACK, SKINLESS, BONELESS, INTERLEAVED",
    "WOOD SAWN OR CHIPPED LENGTHWISE, SLICED OR PEELED, OF CONIFEROUS SPECIES, KILN DRIED",
    "REFINED SUNFLOWER-SEED OIL, WINTERISED, FOOD GRADE QUALITY",
    "FERROUS WASTE AND SCRAP, REMELTING SCRAP INGOTS OF IRON OR STEEL, HMS 1 AND 2 MIXED",
    "COTTON, NOT CARDED OR COMBED, MIDDLING GRADE, STAPLE LENGTH 28 MM AND ABOVE",
    "NATURAL RUBBER TECHNICALLY SPECIFIED TSR20, IN PRIMARY FORMS OR IN PLATES",

    # --- XXL ALL-CAPS trade specifications (121+ chars)
    "STAINLESS STEEL COLD-ROLLED COIL GRADE 304 2B FINISH, THICKNESS 0.5 MM TO 3.0 MM, WIDTH 1000/1219/1500 MM, MILL EDGE, JIS G4305 STANDARD",
    "WHITE REFINED CANE SUGAR ICUMSA 45 RAL MAXIMUM, POLARISATION 99.80 DEGREES MINIMUM, MOISTURE 0.04 PERCENT MAXIMUM, CROP 2025",
    "PRIME NEWSPRINT PAPER IN REELS, GRAMMAGE 45 GSM, REEL WIDTH 762 MM, BRIGHTNESS 60 PERCENT ISO, MOISTURE CONTENT 8 PERCENT MAXIMUM",
    "DOUBLE-REFINED BLEACHED DEODORISED COCONUT OIL, FREE FATTY ACID 0.1 PERCENT MAXIMUM, IODINE VALUE 7.5 TO 10.5, FOOD GRADE QUALITY",
    "GRANULAR UREA 46 PERCENT NITROGEN MINIMUM, BIURET 1 PERCENT MAXIMUM, MOISTURE 0.5 PERCENT MAXIMUM, TREATED WITH ANTI-CAKING AGENT",

    # --- trade abbreviations + spelled-out counterparts (telex/broker style)
    "CPO",
    "crude palm oil",
    "RBD palm olein",
    "palm olein",
    "PKO",
    "crude palm kernel oil",
    "MEG",
    "PTA",
    "purified terephthalic acid",
    "PX",
    "paraxylene",
    "HRC",
    "CRC",
    "SBM",
    "soybean meal",
    "DDGS",
    "distillers dried grains with solubles",
    "FFB",
    "fresh fruit bunches",
    "UCO",
    "used cooking oil",
    "ULSD",
    "ULSD 10ppm diesel",
    "DRI",
    "direct reduced iron",
    "HBI",
    "hot briquetted iron",
    "PET resin",
    "PVC resin",
    "SMP",
    "skimmed milk powder",
    "WMP",
    "whole milk powder",

    # --- trade grades with spec codes inside the span (§8.1 includes codes)
    "ICUMSA 45 sugar",
    "white sugar ICUMSA 45",
    "raw sugar ICUMSA 600-1200",
    "cement CEM I 42.5N",
    "ordinary Portland cement 42.5N",
    "A4 copy paper 80gsm",
    "kraft liner board 175gsm",
    "cement clinker",
    "sawn timber",
    "kiln-dried sawn timber",

    # --- brand / trade names as part of the commodity (§4.8)
    "Nescafé instant coffee",
    "Coca-Cola concentrate syrup",
    "Nestlé milk chocolate bars",
    "Apple iPhone 15 handsets",
    "Samsung Galaxy smartphones",
    "Toyota Camry sedans",
    "Dangote Portland cement",
    "Häagen-Dazs ice cream",
    "Patagonia down jackets",
    "Levi's denim jeans",
    "Nike athletic footwear",
    "Michelin radial truck tyres",
    "Tata steel billets",
    "Shell Rimula engine oil",
    "Yamaha outboard motors",

    # --- dangerous-goods declarations with inline UN numbers (the UN number
    # --- sits inside the noun phrase; standalone class metadata is the
    # --- `dg_class` decoy pool instead)
    "SULPHURIC ACID UN 1830",
    "sulphuric acid UN 1830",
    "SODIUM HYDROXIDE SOLUTION UN 1824",
    "LITHIUM ION BATTERIES UN 3480",
    "lithium ion batteries UN 3480",
    "AMMONIUM NITRATE UN 1942",
    "FIREWORKS UN 0336",
    "CALCIUM HYPOCHLORITE UN 1748",
    "ISOPROPANOL UN 1219",
    "PAINT, FLAMMABLE LIQUID UN 1263",
    "ENVIRONMENTALLY HAZARDOUS SUBSTANCE, LIQUID, N.O.S. UN 3082",

    # --- un- prefixed product qualifiers (POS despite the negative look)
    "unrefined shea butter",
    "unbleached kraft paper",
    "unwashed raw wool",
    "uncoated woodfree paper",
    "undyed cotton fabric",

    # --- extra XS / short heads (families + short-span coverage)
    "resin",
    "paper",
    "fuel",
    "yarn",
    "wine",
    "machinery",
    "fibre",
    "milk powder",
    "ore",
    "tea",
    "rye",
    "hay",
    "salt",
    "jute",
    "hides",
    "scrap",
    "clinker",
    "gypsum",
    "barite",
    "timber",
    "steel coil",
    "steel coils",
    "copper cathodes",
    "shrimp",
    "frozen shrimp",
    "batteries",
    "tyres",
    "vehicles",

    # --- ALL-CAPS surfaces of high-frequency commodities (telex-style B/L
    # --- casing; dedup is case-sensitive so both casings ship)
    "ROBUSTA COFFEE",
    "ARABICA COFFEE",
    "RAW CANE SUGAR",
    "WHITE REFINED SUGAR",
    "CRUDE PALM OIL",
    "STEEL COILS",
    "HOT-ROLLED STEEL COIL",
    "COLD-ROLLED STEEL COIL",
    "COPPER CATHODES",
    "ALUMINIUM INGOTS",
    "PORTLAND CEMENT",
    "GRANULAR UREA",
    "YELLOW CORN",
    "SOYBEAN MEAL",
    "WHEAT FLOUR",
    "BASMATI RICE",
    "LONG GRAIN WHITE RICE",
    "FROZEN CHICKEN",
    "FROZEN SHRIMP",
    "COTTON YARN",
    "PET RESIN",
    "HDPE RESIN",
    "CAUSTIC SODA",
    "ANHYDROUS AMMONIA",
    "GENERAL CARGO",
    "PERSONAL EFFECTS",
    "HOUSEHOLD GOODS",
    "USED MACHINERY",
    "SPARE PARTS",
    "SAWN TIMBER",
    "CEMENT CLINKER",
    "SCRAP METAL",

]
