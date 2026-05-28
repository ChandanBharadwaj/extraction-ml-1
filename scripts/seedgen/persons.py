"""
PERSON entity pool for synthetic NER training-data seed generation.

Coverage follows the data specification:
  - §4.1  PERSON span boundary rules (titles excluded, particles included,
          generational suffixes included when no separating comma)
  - §4.5  Regional name patterns (~60 culture/region rows)
  - §4.6  Honorifics/credentials: NOT included in any span here
  - §4.9  Fuzzy PERSON variants (typos, OCR corruption, diacritic drops,
          casing variants, apostrophe variants, initial-only, en/em-dash)
  - §4.9  Truncation patterns (mid-word and ellipsis cuts)
  - §11.1 PERSON name diversity minimums (≥30 Western European, ≥30 East
          Asian, ≥30 South Asian, ≥20 MENA, ≥20 sub-Saharan African,
          ≥25 Latin American, ≥10 Pacific Islander)

Each string is the EXACT gold span text — no leading titles/honorifics,
no trailing credentials. Apostrophes in names use double-quoted Python
string literals. No imports, no SQL.
"""

PERSONS: list[str] = [

    # ── Anglo (US / UK / AU / NZ / CA) ───────────────────────────────────────
    "John Robert Smith",
    "Emily Clarke",
    "William James Harper",
    "Charlotte Webb",
    "James Thornton",
    "Olivia Bennett",
    "Thomas Edward Morrison",
    "Grace Whitfield",
    "Henry Ford III",           # generational suffix, no comma → included
    "Arthur Pemberton Jr.",
    "Robert H. Lawson",
    "Margaret Anne Sutherland",
    "George Blackwood",
    "Sarah Louise Kingsley",
    "Edward Montague",
    "Victoria Ashford",
    "R H Macy",                 # initials without periods
    "J. K. Rowling",            # initials with spaced periods

    # ── Irish / Scottish ─────────────────────────────────────────────────────
    "O'Brien",                  # mononym / surname-only form
    "Seamus O'Connell",
    "Fionnuala O'Donnell",
    "Alistair MacDonald",
    "Catriona MacLeod",
    "Declan McCarthy",
    "Siobhan Gallagher",
    "Ciarán Ó Briain",
    "Mac an Bhaird",            # Gaelic particle family name as standalone

    # ── French ───────────────────────────────────────────────────────────────
    "Jean-Paul de la Croix",
    "Marie-Claire Dupont",
    "François Leclerc",
    "Isabelle du Bois",
    "Pierre-Henri Lefebvre",
    "Cécile Moreau",
    "Étienne de Montfort",
    "Nathalie Beaumont",

    # ── Spanish ──────────────────────────────────────────────────────────────
    "Juan García López",
    "María Muñoz",
    "Carlos del Río Herrera",
    "Sofía de la Vega Ruiz",
    "Alejandro Fernández Torres",
    "Lucía Ramírez Ortega",
    "Miguel Ángel Sánchez Pérez",
    "Elena Jiménez Castillo",

    # ── Portuguese (PT / BR) ─────────────────────────────────────────────────
    "João Silva Santos",
    "Luiz Inácio Lula da Silva",
    "Ana Beatriz dos Santos",
    "Pedro Henrique Ferreira",
    "Mariana de Oliveira Costa",
    "António Rodrigues",
    "Beatriz Alves Pereira",
    "Rodrigo Sousa Mendes",

    # ── Italian ──────────────────────────────────────────────────────────────
    "Marco di Stefano",
    "Giulia della Rosa",
    "Luca Esposito",
    "Valentina Ferrari",
    "Antonio dal Molin",
    "Francesca Bianchi",
    "Roberto de Santis",
    "Chiara Lombardi",

    # ── German / Austrian / Swiss-German ─────────────────────────────────────
    "Hans Müller",
    "Maria von Habsburg",
    "Friedrich Schäfer",
    "Ursula Bäcker",
    "Wolfgang von der Heide",
    "Ingrid Weiß",
    "Klaus Günther Strauß",
    "Hildegard Zöllner",

    # ── Dutch / Flemish ──────────────────────────────────────────────────────
    "Pieter van der Berg",
    "Annelies van den Houten",
    "Jan van het Veld",
    "Marieke de Boer",
    "Dirk van Loon",
    "Liesbeth Vermeulen",

    # ── Swedish / Norwegian / Danish ─────────────────────────────────────────
    "Lars Eriksson",
    "Astrid Hansen",
    "Björn Andersson",
    "Ingrid Larsen",
    "Sven Christensen",
    "Karin Lindqvist",
    "Ole Pedersen",
    "Annika Nilsson",

    # ── Icelandic ────────────────────────────────────────────────────────────
    "Björk Guðmundsdóttir",
    "Magnús Ólafsson",
    "Sigríður Jónsdóttir",
    "Gunnar Sigurðsson",

    # ── Finnish ──────────────────────────────────────────────────────────────
    "Mika Häkkinen",
    "Sanna Marin",
    "Juhani Virtanen",
    "Aino Mäkinen",
    "Pekka Leinonen",

    # ── Polish ───────────────────────────────────────────────────────────────
    "Wojciech Szczęsny",
    "Anna Kowalska",
    "Piotr Wiśniewski",
    "Katarzyna Wójcik",
    "Tomasz Kowalczyk",
    "Małgorzata Zielińska",

    # ── Czech / Slovak ───────────────────────────────────────────────────────
    "Karel Novák",
    "Marie Nováková",
    "Jakub Procházka",
    "Lucie Horáčková",
    "Marek Blaho",

    # ── Hungarian ────────────────────────────────────────────────────────────
    "Béla Bartók",              # Western order (family+given in HU source; labeled Western here)
    "Erzsébet Kovács",
    "István Szabó",
    "Katalin Varga",

    # ── Russian ──────────────────────────────────────────────────────────────
    "Vladimir Ilyich Lenin",
    "Natalia Sergeyevna Ivanova",
    "Mikhail Alexandrovich Petrov",
    "Olga Dmitrievna Sokolova",
    "Владимир Путин",           # Cyrillic native script
    "Анна Каренина",

    # ── Ukrainian ────────────────────────────────────────────────────────────
    "Volodymyr Zelenskyy",
    "Oksana Petrenko",
    "Mykola Kovalenko",
    "Iryna Marchenko",

    # ── Greek ────────────────────────────────────────────────────────────────
    "Konstantinos Papadopoulos",
    "Eleni Georgiou",
    "Nikos Alexiou",
    "Maria Papakonstantinou",

    # ── Romanian ─────────────────────────────────────────────────────────────
    "Ion Popescu",
    "Elena Ionescu",
    "Gheorghe Dumitrescu",
    "Ioana Marinescu",

    # ── Bulgarian / Serbian / Croatian ───────────────────────────────────────
    "Goran Bregović",
    "Ivan Petrov",
    "Ana Đorđević",
    "Zoran Milanović",
    "Marija Kovačević",

    # ── Turkish ──────────────────────────────────────────────────────────────
    "Mustafa Kemal Atatürk",
    "Fatma Şahin",
    "Mehmet Çelik",
    "Ayşe Kaya",
    "Emre Güneş",

    # ── Hebrew / Israeli ─────────────────────────────────────────────────────
    "David ben Gurion",
    "Tzipi Livni",
    "Yonatan bat Miriam",       # bat (daughter of) particle
    "Moshe Dayan",
    "Golda Meir",

    # ── Arabic — Levantine / Gulf ────────────────────────────────────────────
    "Mohammed bin Salman Al Saud",
    "Fatima bint Khalid Al Rashid",
    "Omar ibn Khattab",
    "Rania Al Abdullah",
    "Abdullah bin Abdulaziz",
    "Layla Al Mansouri",

    # ── Arabic — kunya form ──────────────────────────────────────────────────
    "Abu Mazen",
    "Umm Khalid",
    "Abu Bakr al-Baghdadi",

    # ── Persian / Farsi ──────────────────────────────────────────────────────
    "Mohammad Ali Khamenei",
    "Fatemeh Rahimi",
    "Hossein Rezaei",
    "Shirin Ebadi",
    "Dariush Mehrjui",

    # ── Chinese (Mandarin) ───────────────────────────────────────────────────
    "Wang Wei",
    "Li Na",
    "Zhang Wei",
    "Liu Yang",
    "Chen Jing",
    "毛泽东",                   # Mao Zedong, native Simplified Chinese
    "王伟",                      # Wang Wei, native script
    "习近平",
    "Mao Zedong",               # Pinyin romanization
    "Mao Tse-tung",             # Wade-Giles romanization
    "Li Keqiang",
    "Zhao Lei",

    # ── Chinese (Cantonese / HK) ─────────────────────────────────────────────
    "Wong Ka-Wai",
    "Chan Ho-Man",
    "Lam Cheuk-Ting",
    "Cheung Siu-Fai",

    # ── Japanese ─────────────────────────────────────────────────────────────
    "Yamamoto Tarō",            # family+given (JP order)
    "Tarō Yamamoto",            # Western order
    "山本太郎",                  # native Kanji
    "Keiko Tanaka",
    "Haruki Murakami",
    "Naomi Osaka",
    "Ichiro Suzuki",

    # ── Korean ───────────────────────────────────────────────────────────────
    "Kim Min-jun",
    "Park Ji-sung",
    "Lee Soo-yeon",
    "Choi Dong-wook",
    "Jung Yu-mi",
    "Han Ji-min",

    # ── Vietnamese ───────────────────────────────────────────────────────────
    "Nguyễn Văn Anh",
    "Trần Thị Lan",
    "Lê Minh Tuấn",
    "Phạm Thị Hoa",
    "Hoàng Văn Nam",

    # ── Thai ─────────────────────────────────────────────────────────────────
    "Somchai Sirikhom",
    "Nattaporn Chaisombat",
    "Weerasak Kowsurat",
    "Supatra Masdit",

    # ── Indonesian / Malay ───────────────────────────────────────────────────
    "Sukarno",                  # mononym
    "Joko Widodo",
    "Megawati Sukarnoputri",
    "Ahmad bin Ibrahim",
    "Siti Nurhaliza",
    "Ridwan Kamil",

    # ── Filipino ─────────────────────────────────────────────────────────────
    "Juan Dela Cruz",
    "María Santos y Reyes",
    "Jose Rizal",
    "Corazon Aquino",
    "Gloria Macapagal Arroyo",

    # ── Indian — North (Hindi belt) ──────────────────────────────────────────
    "Aarav Kumar",
    "Priya Sharma",
    "Rahul Gupta",
    "Neha Verma",
    "Amit Singh",
    "Sunita Mishra",
    "Vikram Tiwari",
    "Pooja Yadav",

    # ── Indian — South Tamil ─────────────────────────────────────────────────
    "A.R. Rahman",
    "M. Karunanidhi",
    "S. Janaki",
    "R. Ashwin",
    "T. Rajendran",

    # ── Indian — South Telugu / Kannada / Malayalam ──────────────────────────
    "K. Chandrasekhara Rao",
    "M. K. Stalin",
    "Pinarayi Vijayan",
    "Y. S. Jagan Mohan Reddy",
    "Siddaramaiah",             # mononym (Kannada political figure)

    # ── Indian — Sikh ────────────────────────────────────────────────────────
    "Harpreet Singh",
    "Simran Kaur Sandhu",
    "Gurpreet Singh Dhaliwal",
    "Manpreet Kaur",

    # ── Indian — Bengali ─────────────────────────────────────────────────────
    "Soumya Chakraborty",
    "Rabindranath Tagore",
    "Amartya Sen",
    "Mamata Banerjee",
    "Subhas Chandra Bose",

    # ── Indian — Maharashtrian ───────────────────────────────────────────────
    "Sachin Ramesh Tendulkar",
    "Bal Gangadhar Tilak",
    "Lata Mangeshkar",
    "Sunil Gavaskar",

    # ── Indian — Gujarati / Parsi ────────────────────────────────────────────
    "Narendra Modi",
    "Ratan Tata",
    "Dhirubhai Ambani",
    "Kalpana Chawla",

    # ── Pakistani ────────────────────────────────────────────────────────────
    "Muhammad Ali Khan",
    "Benazir Bhutto",
    "Imran Khan",
    "Malala Yousafzai",
    "Asif Ali Zardari",

    # ── Sri Lankan Sinhala ───────────────────────────────────────────────────
    "D.M. Jayaratne",
    "Mahinda Rajapaksa",
    "Chandrika Kumaratunga",
    "Ranil Wickremasinghe",

    # ── Sri Lankan Tamil ─────────────────────────────────────────────────────
    "V. Prabhakaran",
    "S. J. V. Chelvanayakam",

    # ── Bangladeshi ──────────────────────────────────────────────────────────
    "Sheikh Hasina",
    "Md. Karim",                # Md. abbreviation included per §4.9
    "Muhammad Yunus",
    "Khaleda Zia",

    # ── Nepalese / Bhutanese ─────────────────────────────────────────────────
    "Pushpa Kamal Dahal",
    "Karma Lhamo",
    "Girija Prasad Koirala",
    "Tenzin Dorji",

    # ── Yoruba (Nigeria) ─────────────────────────────────────────────────────
    "Adeyemi Adebayo",
    "Wole Soyinka",
    "Ngozi Adeyemi",
    "Oluwaseun Akinwale",
    "Adebimpe Oluwafemi",

    # ── Igbo (Nigeria) ───────────────────────────────────────────────────────
    "Chukwuemeka Ojukwu",
    "Chinua Achebe",
    "Ngozi Okonjo-Iweala",
    "Chidi Okeke",
    "Adaeze Nwosu",

    # ── Hausa ────────────────────────────────────────────────────────────────
    "Mohammed Buhari",
    "Aliyu Wamakko",
    "Aminu Kano",
    "Fatima Aliyu",

    # ── Ethiopian / Eritrean ─────────────────────────────────────────────────
    "Abebe Bikila",
    "Haile Selassie",
    "Tewodros Adhanom Ghebreyesus",
    "Abiy Ahmed Ali",
    "Tirunesh Dibaba",

    # ── Somali ───────────────────────────────────────────────────────────────
    "Mohammed Abdullahi Mohammed",
    "Fadumo Dayib",
    "Hassan Sheikh Mohamud",

    # ── Swahili (East Africa) ────────────────────────────────────────────────
    "Uhuru Kenyatta",
    "Julius Nyerere",
    "Wangari Maathai",
    "Paul Kagame",
    "Jakaya Kikwete",

    # ── Afrikaans (South Africa) ─────────────────────────────────────────────
    "Jan van der Merwe",
    "Pieter du Toit",
    "Annetjie van Rensburg",
    "Hendrik Verwoerd",

    # ── Zulu / Xhosa / Sotho ─────────────────────────────────────────────────
    "Nelson Rolihlahla Mandela",
    "Cyril Ramaphosa",
    "Desmond Tutu",
    "Miriam Makeba",
    "Thabo Mbeki",

    # ── West African Francophone ─────────────────────────────────────────────
    "Aminata Diallo",
    "Aliou Touré",
    "Ousmane Sembène",
    "Fatou Diop",
    "Moussa Cissé",
    "Aissatou Ba",

    # ── Mexican ──────────────────────────────────────────────────────────────
    "Carlos García Rodríguez",
    "Frida Kahlo y Calderón",
    "Andrés Manuel López Obrador",
    "Sor Juana Inés de la Cruz",
    "Cuauhtémoc Cárdenas Solórzano",
    "Dolores Huerta",

    # ── Argentinian / Uruguayan ──────────────────────────────────────────────
    "Diego Armando Maradona",
    "Lionel Andrés Messi",
    "Jorge Luis Borges",
    "Astor Pantaleón Piazzolla",
    "Eva María Duarte de Perón",

    # ── Brazilian ────────────────────────────────────────────────────────────
    "Pelé",                       # mononym
    "Ayrton Senna da Silva",
    "Dilma Vana Rousseff",
    "Neymar da Silva Santos Júnior",
    "Fernanda Montenegro",

    # ── Cuban / Caribbean ────────────────────────────────────────────────────
    "Pedro Pérez Cabrera",
    "Celia Cruz",
    "Fidel Alejandro Castro Ruz",
    "Alicia Alonso",

    # ── Indigenous Mesoamerican ──────────────────────────────────────────────
    "Cuauhtémoc",               # mononym (last Aztec emperor)
    "Atahualpa",                # mononym (Inca ruler)
    "Rigoberta Menchú Tum",
    "Evo Morales Ayma",

    # ── Hawaiian ─────────────────────────────────────────────────────────────
    "Kamehameha",               # mononym
    "Liliʻuokalani",            # okina in name
    "Bernice Pauahi Bishop",
    "Daniel Kahanamoku",

    # ── Maori (NZ) ───────────────────────────────────────────────────────────
    "Tāmati Coffey",
    "Witi Ihimaera",
    "Hone Harawira",
    "Jacinda Kate Laurell Ardern",

    # ── Polynesian (Samoan, Tongan, Fijian) ──────────────────────────────────
    "Sione Filitonga",
    "Tupou Vaipulu",
    "Iosefa Solofa",
    "Asenati Lole-Taylor",
    "Sitiveni Rabuka",

    # ── Mixed-heritage / diasporic ───────────────────────────────────────────
    "Mary Chen-Smith",
    "Lily Wong (Wang Lihua)",   # romanization + native parens
    "David Kim-Nakamura",
    "Priya Patel-Johnson",
    "Sophie Nguyen-Martin",

    # ── Names that double as common nouns ────────────────────────────────────
    "Mark",                     # mononym / common noun
    "Will",
    "Hope",
    "Grace",
    "Faith",
    "Joy Okonkwo",

    # ── Names that double as ORG surface forms (§4.1 adversarial) ────────────
    "Tesla",                    # context: role=signer → PERSON span
    "Ford",
    "Hermès Lopez",
    "Edison Clarke",

    # ── Structural variety: hyphenated ───────────────────────────────────────
    "Mary-Kate Olsen",
    "Anne-Marie Duplessis",
    "Smith-Jones",
    "Jean-Baptiste Leroy",
    "Li-Wei Zhang",

    # ── Structural variety: apostrophes ──────────────────────────────────────
    "D'Souza",
    "D'Arcy Thompson",
    "O'Connell",
    "L'Heureux",

    # ── Structural variety: particles ────────────────────────────────────────
    "von Neumann",              # surname-only with particle
    "de la Cruz",

    # ── Structural variety: all-lowercase (OCR / fast-typed) ─────────────────
    "maria gonzalez",
    "john smith",
    "carlos garcia",
    "priya sharma",

    # ── Structural variety: ALL-CAPS ─────────────────────────────────────────
    "MARIA GONZALEZ",
    "JOHN SMITH",
    "CARLOS GARCIA",
    "ZHANG WEI",

    # ── Structural variety: initials ─────────────────────────────────────────
    "T. S. Eliot",
    "W. E. B. Du Bois",
    "S. Hansen",                # initial-only per §4.9
    "M. Muñoz",
    "Felix Y",                  # truncation pattern §4.9

    # ── Devanagari / non-Latin scripts ───────────────────────────────────────
    "महात्मा गांधी",             # Devanagari: Mahatma Gandhi
    "सचिन तेंदुलकर",             # Devanagari: Sachin Tendulkar
    "محمد علي",                  # Arabic: Muhammad Ali
    "日立太郎",                    # Japanese Kanji (additional script coverage form)
    "이순신",                     # Korean: Yi Sun-sin

    # ── Fuzzy / OCR variants from §4.9 ───────────────────────────────────────
    "Maria Gonzlaez",           # transposition typo
    "Mria Gonzalez",            # dropped character
    "Ma1ia Gonzalez",           # OCR 1 ↔ r confusion
    "Soren Hansen",             # diacritic dropped (Søren)
    "SOREN HANSEN",             # all-caps + diacritic dropped
    "Maria Munoz",              # diacritic dropped (María Muñoz)
    "MARIA MUNOZ",              # all-caps + diacritics dropped
    "Maria Munhoz",             # typo variant (Muñoz → Munhoz)
    "JK Rowling",               # no periods on initials
    "J K Rowling",              # space-only initials
    "J.K Rowling",              # missing final period
    "J.K.Rowling",              # no spaces between initials and surname
    "OBrien",                   # apostrophe dropped
    "O’Brien",             # curly right single quotation mark apostrophe
    "O Brien",                  # apostrophe replaced by space
    "Mary–Kate",           # en-dash (–) between given names
    "Mary—Kate",           # em-dash (—) between given names
    "MaryKate",                 # separator removed entirely
    "Mary Kate",                # hyphen replaced by space
    "MAO ZEDONG",               # all-caps Pinyin
    "MariaGonzalez",            # whitespace dropped
    "Maria  Gonzalez",          # double-space

    # ── Truncation patterns from §4.9 ────────────────────────────────────────
    "Maria Gonz",               # mid-word truncation
    "Nguyễn Văn",               # mid-name truncation
    "Konstantinos Papadopo",    # mid-word family-name truncation

    # ── Mononyms (§8.3) ──────────────────────────────────────────────────────
    "Madonna",
    "Björk",
    "Prince",
    "Cher",
    "Adele",
    "Zendaya",

]
