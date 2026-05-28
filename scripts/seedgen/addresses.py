"""ADDRESS entity pool for the synthetic NER training-data seed.

Contract references (docs/data_specification.md):
  §4.3   ADDRESS surface forms (single-line, multi-line, PO Box, suite/floor,
         country token, US/UK/Asian/industrial formats, non-ASCII, truncated,
         ALL-CAPS, mixed casing).
  §8.2   ADDRESS boundary discipline: include fullest contiguous locational
         string (street number through country); exclude leading prepositions
         and descriptors; multi-line is one span; apt/suite is part of span.
  §3.1   Length buckets: S (4–10), M (11–25), L (26–60), XL (61–120),
         XXL (121+). ADDRESS skews M/L/XL with meaningful XL and XXL counts.
  §3.2   Minimum support per bucket: S≥10 (PO boxes), M≥30, L≥80, XL≥70,
         XXL≥30.
  §11.2  ADDRESS country diversity: US, UK, EU (Germany/France/Netherlands/
         Italy/Spain/etc.), India, China, Japan, Singapore, Australia, Brazil,
         Mexico, South Africa, Saudi Arabia, UAE — each ≥15 records.

Every value is the EXACT gold span text. No leading prepositions.
Plain Python strings; downstream builder handles SQL escaping.
Values containing apostrophes use double-quote delimiters.
Newlines encoded as real \\n inside multi-line strings.
"""

ADDRESSES: list[str] = [

    # =========================================================================
    # UNITED STATES  (≥15 records)
    # Formats: "1234 Name St, City, ST ZIPCODE" and variants
    # =========================================================================

    # S-bucket: minimal ZIP-only or abbreviated
    "Chicago, IL",

    # M-bucket
    "1 Infinite Loop, Cupertino",
    "100 Main St, Boston, MA",

    # L-bucket — standard US format, single line
    "1234 Main St, Springfield, IL 62701",
    "350 Fifth Avenue, New York, NY 10118",
    "1600 Pennsylvania Ave NW, Washington, DC 20500",
    "200 Park Ave, New York, NY 10166",
    "101 California St, San Francisco, CA 94111",
    "1 Hacker Way, Menlo Park, CA 94025",
    "410 Terry Ave N, Seattle, WA 98109",
    "500 W Madison St, Chicago, IL 60661",
    "2200 Mission College Blvd, Santa Clara, CA 95054",
    "10960 Wilshire Blvd, Los Angeles, CA 90024",
    "7000 Marina Blvd, Brisbane, CA 94005",
    "Bldg 40, 1 Microsoft Way, Redmond, WA 98052",

    # L-bucket — with suite / unit
    "Suite 300, 1050 Connecticut Ave NW, Washington, DC 20036",
    "Floor 12, 225 Bush St, San Francisco, CA 94104",
    "Unit 4B, 120 Water St, New York, NY 10005",
    "Apt 7C, 302 W 86th St, New York, NY 10024",

    # L-bucket — ALL-CAPS (OCR'd letterhead)
    "1234 MAIN ST, SPRINGFIELD, IL 62701",
    "350 FIFTH AVENUE, NEW YORK, NY 10118",

    # XL-bucket — full with country
    "1234 Main St, Springfield, IL 62701, United States",
    "350 Fifth Avenue, New York, NY 10118, United States of America",
    "Suite 300, 1050 Connecticut Ave NW, Washington, DC 20036, USA",

    # XL multi-line
    "1 Infinite Loop\nCupertino, CA 95014\nUnited States",
    "350 Fifth Avenue\nNew York, NY 10118\nUSA",

    # XXL — care-of / very long multi-line
    "c/o Global Logistics Inc.\n1234 Industrial Park Drive, Suite 200\nChicago, IL 60601\nUnited States of America",
    "Attn: Receiving Dept\n2200 Mission College Blvd, Building C\nSanta Clara, CA 95054\nUnited States",

    # PO Box — US
    "PO Box 4567, Houston, TX 77001",
    "P.O. Box 1099, Miami, FL 33101",

    # truncated / OCR-cut
    "1234 Main St, Springfiel",

    # =========================================================================
    # UNITED KINGDOM  (≥15 records)
    # Formats: "Number Name Street, City POSTCODE" and variants
    # =========================================================================

    # S
    "London EC2V 8RF",

    # M
    "10 Downing Street, London",
    "1 Churchill Place, London",

    # L
    "10 Downing Street, London SW1A 2AA",
    "1 Churchill Place, Canary Wharf, London E14 5HP",
    "30 St Mary Axe, London EC3A 8BF",
    "1 London Bridge St, London SE1 9GF",
    "100 Cheapside, London EC2V 6DT",
    "Buckingham Palace Road, London SW1W 0QH",
    "12 High Street, Manchester M1 1PT",
    "1 Deansgate, Manchester M3 3FF",
    "5 Brindleyplace, Birmingham B1 2JB",

    # L — with country
    "10 Downing Street, London SW1A 2AA, United Kingdom",
    "30 St Mary Axe, London EC3A 8BF, United Kingdom",
    "1 Churchill Place, Canary Wharf, London E14 5HP, UK",

    # L — ALL-CAPS
    "10 DOWNING STREET, LONDON SW1A 2AA, UNITED KINGDOM",
    "100 CHEAPSIDE, LONDON EC2V 6DT, UNITED KINGDOM",

    # XL multi-line
    "10 Downing Street\nLondon\nSW1A 2AA\nUnited Kingdom",
    "1 Churchill Place\nCanary Wharf\nLondon E14 5HP\nUnited Kingdom",
    "30 St Mary Axe\nLondon\nEC3A 8BF",

    # XXL — care-of multi-line
    "c/o BNP Paribas UK Limited\n10 Harewood Avenue\nLondon NW1 6AA\nUnited Kingdom",

    # PO Box — UK
    "PO Box 100, Edinburgh EH3 5WW",

    # Suite / floor
    "Floor 6, 1 London Wall Place, London EC2Y 5AU",

    # Truncated
    "10 Downing Street, Londo",

    # =========================================================================
    # GERMANY  (≥15 records, EU)
    # Formats: "Straßenname Hausnummer, PLZ Stadt"
    # =========================================================================

    # M
    "Maximilianstraße 2, München",
    "Unter den Linden 1, Berlin",

    # L
    "Maximilianstraße 2, 80539 München",
    "Unter den Linden 1, 10117 Berlin",
    "Potsdamer Platz 1, 10785 Berlin",
    "Bockenheimer Landstraße 24, 60323 Frankfurt am Main",
    "Domkloster 4, 50667 Köln",
    "Mönckebergstraße 7, 20095 Hamburg",
    "Königstraße 5, 70173 Stuttgart",
    "Bahnhofsplatz 1, 90402 Nürnberg",
    "Beethovenstraße 12, 80336 München",

    # L — with country
    "Maximilianstraße 2, 80539 München, Deutschland",
    "Potsdamer Platz 1, 10785 Berlin, Germany",
    "Domkloster 4, 50667 Köln, Germany",

    # XL multi-line — diacritics
    "Bockenheimer Landstraße 24\n60323 Frankfurt am Main\nDeutschland",
    "Potsdamer Platz 1\n10785 Berlin\nGermany",
    "Mönckebergstraße 7\n20095 Hamburg\nGermany",

    # XXL care-of
    "z. Hd. Einkaufsabteilung\nBockenheimer Landstraße 24\n60323 Frankfurt am Main\nDeutschland",

    # =========================================================================
    # FRANCE  (≥15 records, EU)
    # Formats: "N° Rue Nom, Code Ville"
    # =========================================================================

    # M
    "8 Avenue de l'Opéra, Paris",
    "1 Place du Parvis Notre-Dame",

    # L
    "8 Avenue de l'Opéra, 75001 Paris",
    "15 Rue de Rivoli, 75001 Paris",
    "1 Place du Parvis Notre-Dame, 75004 Paris",
    "10 Rue de la Paix, 75002 Paris",
    "Tour Eiffel, Champ de Mars, 75007 Paris",
    "5 Rue d'Aboukir, 75002 Paris",
    "22 Cours du Médoc, 33300 Bordeaux",
    "3 Cours Mirabeau, 13100 Aix-en-Provence",
    "1 Allée Paul Sabatier, 31000 Toulouse",

    # L — with country
    "8 Avenue de l'Opéra, 75001 Paris, France",
    "15 Rue de Rivoli, 75001 Paris, France",
    "22 Cours du Médoc, 33300 Bordeaux, France",

    # XL multi-line
    "8 Avenue de l'Opéra\n75001 Paris\nFrance",
    "10 Rue de la Paix\n75002 Paris\nFrance",

    # XXL
    "À l'attention du service logistique\n15 Rue de Rivoli\n75001 Paris\nFrance",

    # =========================================================================
    # NETHERLANDS  (≥15 records, EU)
    # Formats: "Straatnaam HuisNr, PostCode Stad"
    # =========================================================================

    # M
    "Herengracht 420, Amsterdam",
    "Coolsingel 40, Rotterdam",

    # L
    "42 Industrial Park Road, Rotterdam, 3011 AB",
    "Herengracht 420, 1017 BZ Amsterdam",
    "Coolsingel 40, 3011 AD Rotterdam",
    "Weena 70, 3012 CM Rotterdam",
    "Wilhelminaplein 1, 3072 DE Rotterdam",
    "Gustav Mahlerplein 200, 1082 MS Amsterdam",
    "Bezuidenhoutseweg 12, 2594 AV Den Haag",
    "Laan van NOI 5, 2595 GA Den Haag",
    "Koningin Julianaplein 30, 2595 AA Den Haag",

    # L — with country
    "Herengracht 420, 1017 BZ Amsterdam, Netherlands",
    "Weena 70, 3012 CM Rotterdam, The Netherlands",

    # XL multi-line
    "42 Industrial Park Road\nRotterdam\n3011 AB",
    "42 Industrial Park Road\nRotterdam\n3011 AB\nNetherlands",
    "Herengracht 420\n1017 BZ Amsterdam\nNederland",

    # XXL
    "t.a.v. Logistieke Afdeling\nWeena 70\n3012 CM Rotterdam\nNederland",

    # Truncated OCR (from spec example)
    "42 Industrial Park Road, Rotterda",

    # =========================================================================
    # ITALY  (≥15 records, EU)
    # Formats: "Via/Piazza Nome N°, CAP Città (Prov)"
    # =========================================================================

    # M
    "Via Roma 15, Milano",
    "Piazza del Colosseo, Roma",

    # L
    "Via Roma 15, 20121 Milano",
    "Piazza del Colosseo 1, 00184 Roma",
    "Corso Buenos Aires 30, 20124 Milano",
    "Via Toledo 256, 80134 Napoli",
    "Via della Vigna Nuova 18, 50123 Firenze",
    "Corso Vittorio Emanuele II 52, 10125 Torino",
    "Via XX Settembre 33, 16121 Genova",
    "Via Maqueda 100, 90134 Palermo",

    # L — with country
    "Via Roma 15, 20121 Milano, Italia",
    "Via della Vigna Nuova 18, 50123 Firenze, Italy",
    "Via Toledo 256, 80134 Napoli, Italy",

    # XL multi-line
    "Via Roma 15\n20121 Milano\nItalia",
    "Piazza del Colosseo 1\n00184 Roma\nItaly",

    # XXL
    "c/o Acme Italia S.p.A.\nCorso Buenos Aires 30\n20124 Milano\nItalia",

    # =========================================================================
    # SPAIN  (≥15 records, EU)
    # Formats: "Calle/Avenida Nombre N°, CP Ciudad (Prov)"
    # =========================================================================

    # M
    "Calle Mayor 1, Madrid",
    "Las Ramblas 92, Barcelona",

    # L
    "Calle Mayor 1, 28013 Madrid",
    "Las Ramblas 92, 08001 Barcelona",
    "Gran Vía 32, 28013 Madrid",
    "Passeig de Gràcia 43, 08007 Barcelona",
    "Avenida de la Constitución 1, 41004 Sevilla",
    "Calle Alcalá 21, 28014 Madrid",
    "Calle Marqués de Larios 6, 29005 Málaga",
    "Calle de Pelayo 12, 28004 Madrid",

    # L — with country
    "Calle Mayor 1, 28013 Madrid, España",
    "Passeig de Gràcia 43, 08007 Barcelona, Spain",
    "Calle Marqués de Larios 6, 29005 Málaga, Spain",

    # XL multi-line
    "Gran Vía 32\n28013 Madrid\nEspaña",
    "Calle Marqués de Larios 6\n29005 Málaga\nSpain",

    # XXL
    "A/A Departamento de Logística\nCalle Alcalá 21\n28014 Madrid\nEspaña",

    # =========================================================================
    # INDIA  (≥15 records)
    # Formats vary: Plot/flat, sector, city, state PIN; industrial park style
    # =========================================================================

    # S
    "Mumbai 400001",

    # M
    "Plot 42, Sector 5, Gurugram",
    "Nariman Point, Mumbai",

    # L
    "Plot 42, Sector 5, Gurugram, 122001",
    "1st Floor, Nirlon Knowledge Park, Mumbai 400063",
    "7th Floor, DLF Cyber City, Phase II, Gurugram, 122002",
    "Flat No 5, Sector 21, Noida, Uttar Pradesh 201301",
    "22 Maker Chambers VI, Nariman Point, Mumbai 400021",
    "No. 44, Lavelle Road, Bengaluru, Karnataka 560001",
    "1 HAL Airport Road, Bengaluru 560008",
    "Plot 14, Rajiv Gandhi Infotech Park, Pune 411057",
    "Infosys Campus, Electronic City Phase 2, Bengaluru 560100",
    "12B, Industrial Area Phase 1, Chandigarh 160002",

    # L — with country
    "Plot 42, Sector 5, Gurugram, 122001, India",
    "22 Maker Chambers VI, Nariman Point, Mumbai 400021, India",
    "No. 44, Lavelle Road, Bengaluru, Karnataka 560001, India",

    # XL multi-line
    "Plot 42, Sector 5\nGurugram, Haryana\n122001, India",
    "22 Maker Chambers VI\nNariman Point\nMumbai 400021\nIndia",
    "7th Floor, DLF Cyber City\nPhase II, Gurugram\nHaryana 122002\nIndia",

    # XXL — care-of multi-line
    "c/o Tata Sons Pvt. Ltd.\nBombay House, 24 Homi Mody Street\nFort, Mumbai 400001\nMaharashtra, India",

    # PO Box — India
    "PO Box 1050, Chennai, Tamil Nadu 600001",

    # Industrial-park style
    "Survey No. 12/3, MIDC Industrial Area, Andheri East, Mumbai 400093, India",
    "Plot C-2, SIPCOT Industrial Complex, Gummidipoondi, Tamil Nadu 601201",

    # =========================================================================
    # CHINA  (≥15 records)
    # Formats: Province, City, District, Street, Building; postal last or first
    # =========================================================================

    # M
    "Pudong New Area, Shanghai",
    "Chaoyang District, Beijing",

    # L
    "No. 1 Century Avenue, Pudong New Area, Shanghai 200120",
    "No. 55 Andingmen East Road, Dongcheng District, Beijing 100007",
    "No. 388 Middle Huaihai Road, Xuhui District, Shanghai 200031",
    "No. 1 Futian Free Trade Zone, Shenzhen 518048",
    "No. 1600 Jiuzhou Avenue, Zhuhai, Guangdong 519015",
    "Bldg 3, No. 9 Gaoxin 3rd Road, Hi-Tech Zone, Xi'an 710075",
    "No. 500 Yishan Road, Xuhui District, Shanghai 200233",
    "Room 1801, Tower B, No. 1 Shuang Long Road, Chengdu 610023",

    # L — with country
    "No. 1 Century Avenue, Pudong New Area, Shanghai 200120, China",
    "No. 388 Middle Huaihai Road, Xuhui District, Shanghai 200031, China",
    "No. 55 Andingmen East Road, Dongcheng District, Beijing 100007, P.R. China",

    # XL multi-line
    "No. 1 Century Avenue\nPudong New Area\nShanghai 200120\nChina",
    "No. 1600 Jiuzhou Avenue\nZhuhai, Guangdong 519015\nChina",

    # XXL
    "c/o China National Petroleum Corporation\nNo. 9 Dongzhimen North Street\nChaoyang District\nBeijing 100028\nP.R. China",

    # PO Box
    "P.O. Box 100, Shanghai 200001, China",

    # =========================================================================
    # JAPAN  (≥15 records)
    # Postal code format: NNN-NNNN; often before city
    # =========================================================================

    # S
    "Tokyo 100-0001",

    # M
    "100-0001 Chiyoda-ku, Tokyo",
    "Shinjuku-ku, Tokyo",

    # L
    "2-2-1 Uchisaiwaicho, Chiyoda-ku, Tokyo 100-0011",
    "1-1-1 Marunouchi, Chiyoda-ku, Tokyo 100-0005",
    "2-7-2 Marunouchi, Chiyoda-ku, Tokyo 100-0005",
    "1-3-1 Nihonbashi, Chuo-ku, Tokyo 103-0027",
    "2-1-1 Nishi-Shinjuku, Shinjuku-ku, Tokyo 163-8001",
    "1-1 Sakurajima, Konohana-ku, Osaka 554-0031",
    "3-1-1 Minatojima Minamimachi, Chuo-ku, Kobe 650-0047",
    "2-2-1 Kita-Aoyama, Minato-ku, Tokyo 107-8419",

    # L — with country
    "2-2-1 Uchisaiwaicho, Chiyoda-ku, Tokyo 100-0011, Japan",
    "1-1-1 Marunouchi, Chiyoda-ku, Tokyo 100-0005, Japan",
    "1-3-1 Nihonbashi, Chuo-ku, Tokyo 103-0027, Japan",

    # XL multi-line
    "Tokyo 100-0001, Japan",
    "2-2-1 Uchisaiwaicho\nChiyoda-ku\nTokyo 100-0011\nJapan",
    "1-1-1 Marunouchi\nChiyoda-ku\nTokyo 100-0005\nJapan",

    # XXL
    "c/o Mitsubishi Corporation\n2-3-1 Marunouchi\nChiyoda-ku\nTokyo 100-8086\nJapan",

    # PO Box
    "PO Box 15, Tokyo 100-8986, Japan",

    # =========================================================================
    # SINGAPORE  (≥15 records)
    # Formats: "Number Name, Singapore NNNNNN"
    # =========================================================================

    # S
    "Singapore 049320",
    "Singapore 018956",

    # M
    "7 Canal St, Singapore",
    "1 Raffles Quay, Singapore",

    # L
    "7 Canal St, Singapore 049320",
    "1 Raffles Quay, Singapore 048583",
    "8 Marina View, Asia Square Tower 1, Singapore 018960",
    "3 Church Street, Samsung Hub, Singapore 049483",
    "50 Raffles Place, Singapore Land Tower, Singapore 048623",
    "10 Collyer Quay, Ocean Financial Centre, Singapore 049315",
    "30 Pasir Panjang Road, Singapore 117440",
    "9 Straits View, Marina One West Tower, Singapore 018937",
    "80 Robinson Road, Singapore 068898",

    # L — with country
    "7 Canal St, Singapore 049320, Republic of Singapore",
    "1 Raffles Quay, Singapore 048583, Republic of Singapore",
    "80 Robinson Road, Singapore 068898, Republic of Singapore",

    # Suite / floor
    "Level 10, 8 Marina View, Asia Square Tower 1, Singapore 018960",
    "Suite 15-01, 3 Church Street, Samsung Hub, Singapore 049483",

    # XL multi-line
    "8 Marina View\nAsia Square Tower 1\nSingapore 018960",
    "10 Collyer Quay\nOcean Financial Centre\nSingapore 049315",
    "50 Raffles Place\nSingapore Land Tower\nSingapore 048623\nRepublic of Singapore",

    # XXL
    "c/o Singapore Trade Development Board\n230 Victoria Street\nPergola Road\nSingapore 188024",

    # PO Box
    "PO Box 101, Singapore 911401",

    # =========================================================================
    # AUSTRALIA  (≥15 records)
    # Formats: "N Street Name, Suburb STATE Postcode"
    # =========================================================================

    # S
    "Sydney NSW 2000",
    "Melbourne VIC 3000",

    # M
    "1 Martin Place, Sydney",
    "360 Collins St, Melbourne",

    # L
    "1 Martin Place, Sydney NSW 2000",
    "360 Collins St, Melbourne VIC 3000",
    "55 Collins Street, Melbourne VIC 3000",
    "100 King Street, Melbourne VIC 3000",
    "Governor Phillip Tower, 1 Farrer Place, Sydney NSW 2000",
    "Level 10, 20 Martin Place, Sydney NSW 2000",
    "Level 30, 600 Bourke Street, Melbourne VIC 3000",
    "Queen Victoria Building, 455 George Street, Sydney NSW 2000",
    "Westfield Tower 2, 101 Grafton Street, Bondi Junction NSW 2022",

    # L — with country
    "1 Martin Place, Sydney NSW 2000, Australia",
    "360 Collins St, Melbourne VIC 3000, Australia",
    "100 King Street, Melbourne VIC 3000, Australia",

    # XL multi-line
    "1 Martin Place\nSydney NSW 2000\nAustralia",
    "360 Collins St\nMelbourne VIC 3000\nAustralia",

    # XXL
    "c/o BHP Billiton Limited\nBHP Billiton Centre\n171 Collins Street\nMelbourne VIC 3000\nAustralia",

    # PO Box
    "PO Box 1234, Sydney NSW 2000",
    "GPO Box 22, Brisbane QLD 4001",

    # Suite / floor
    "Suite 5, 88 Wallaby Way, Sydney NSW 2000",
    "Level 5, Suite 501, 1 Nicholson Street, Melbourne VIC 3000",

    # =========================================================================
    # BRAZIL  (≥15 records)
    # Formats: "Rua/Av Name, N° – Bairro, Cidade – UF, CEP"
    # =========================================================================

    # S
    "São Paulo, SP",

    # M
    "Av. Paulista, São Paulo",
    "Rua do Ouvidor, Rio de Janeiro",

    # L
    "Av. Paulista 1374, Bela Vista, São Paulo, SP 01310-100",
    "Rua do Ouvidor 98, Centro, Rio de Janeiro, RJ 20040-030",
    "Av. Faria Lima 3500, Itaim Bibi, São Paulo, SP 04538-132",
    "Rua XV de Novembro 300, Centro, Curitiba, PR 80020-310",
    "Av. Getúlio Vargas 1492, Savassi, Belo Horizonte, MG 30112-021",
    "Rua Paraíba 550, Funcionários, Belo Horizonte, MG 30130-140",
    "Av. Afonso Pena 867, Centro, Manaus, AM 69020-120",
    "SBS Quadra 1, Bloco A, Ed. BNDES, Brasília, DF 70076-900",

    # L — with country
    "Av. Paulista 1374, São Paulo, SP 01310-100, Brasil",
    "Av. Faria Lima 3500, Itaim Bibi, São Paulo, SP 04538-132, Brazil",
    "Rua do Ouvidor 98, Rio de Janeiro, RJ 20040-030, Brazil",

    # XL multi-line — diacritics
    "Av. Paulista 1374\nBela Vista\nSão Paulo, SP 01310-100\nBrasil",
    "Av. Faria Lima 3500\nItaim Bibi\nSão Paulo, SP 04538-132\nBrazil",

    # XXL
    "A/C Departamento Comercial\nAv. Paulista 1374, Bela Vista\nSão Paulo, SP 01310-100\nBrasil",
    "c/o Petróleo Brasileiro S.A.\nAv. Henrique Valadares 28\nCentro, Rio de Janeiro, RJ 20231-030\nBrasil",

    # PO Box
    "Caixa Postal 1050, São Paulo, SP 01032-000",

    # =========================================================================
    # MEXICO  (≥15 records)
    # Formats: "Calle N°, Colonia, Ciudad, Estado, CP"
    # =========================================================================

    # M
    "Reforma 222, Ciudad de México",
    "Centro, Monterrey, NL",

    # L
    "Paseo de la Reforma 222, Juárez, Ciudad de México, CDMX 06600",
    "Av. Insurgentes Sur 1605, Benito Juárez, CDMX 03900",
    "Blvd. Adolfo Ruiz Cortines 3000, Lomas Virreyes, CDMX 11000",
    "Lago Zurich 245, Granada, Miguel Hidalgo, CDMX 11529",
    "Av. Vasconcelos 101-Pte., Santa Engracia, San Pedro Garza García, NL 66267",
    "Calzada Presidente Juárez 555, Tlalnepantla, Estado de México 54050",
    "Blvd. Manuel Ávila Camacho 1, Tecamachalco, Naucalpan, Estado de México 53950",

    # L — with country
    "Paseo de la Reforma 222, Juárez, CDMX 06600, México",
    "Av. Insurgentes Sur 1605, Benito Juárez, CDMX 03900, Mexico",
    "Lago Zurich 245, Granada, Miguel Hidalgo, CDMX 11529, Mexico",

    # XL multi-line
    "Paseo de la Reforma 222\nJuárez\nCiudad de México, CDMX 06600\nMéxico",
    "Av. Insurgentes Sur 1605\nBenito Juárez\nCDMX 03900\nMexico",

    # XXL
    "A/A Dirección de Logística\nPaseo de la Reforma 222, Torre B\nJuárez, Ciudad de México\nCDMX 06600\nMéxico",

    # PO Box
    "Apartado Postal 950, Ciudad de México, CDMX 06600",

    # =========================================================================
    # SOUTH AFRICA  (≥15 records)
    # Formats: "N Street Name, Suburb, City, Province, Postcode"
    # =========================================================================

    # M
    "Sandton, Johannesburg",
    "Cape Town City Bowl",

    # L
    "1 Simmonds Street, Johannesburg, 2001",
    "5 Merchant Place, Fredman Drive, Sandton, 2196",
    "200 Main Street, Sandton, Johannesburg, 2146",
    "2 Long Street, Cape Town, Western Cape 8001",
    "11 Diagonal Street, Johannesburg CBD, 2001",
    "135 Rivonia Road, Sandton, Johannesburg, 2196",
    "1 Alice Lane, Sandton, Johannesburg, 2196",
    "30 Baker Street, Rosebank, Johannesburg, 2196",
    "9 Adderley Street, Cape Town, 8001",

    # L — with country
    "1 Simmonds Street, Johannesburg, 2001, South Africa",
    "2 Long Street, Cape Town, Western Cape 8001, South Africa",
    "135 Rivonia Road, Sandton, Johannesburg, 2196, South Africa",

    # XL multi-line
    "1 Simmonds Street\nJohannesburg\n2001\nSouth Africa",
    "2 Long Street\nCape Town\nWestern Cape 8001\nSouth Africa",

    # XXL
    "c/o Standard Bank Group Limited\n9th Floor, 5 Simmonds Street\nJohannesburg, Gauteng 2001\nSouth Africa",

    # PO Box
    "PO Box 50, Cape Town 8000, South Africa",

    # =========================================================================
    # SAUDI ARABIA  (≥15 records)
    # Formats vary: district, street, city, postal
    # =========================================================================

    # M
    "King Fahd Road, Riyadh",
    "Olaya District, Riyadh",

    # L
    "King Fahd Road, Al Olaya, Riyadh 11564",
    "Aramco Road, Dhahran, Eastern Province 31311",
    "Prince Mohammed bin Abdulaziz Road, Riyadh 12214",
    "Al Madinah Al Munawarah Road, Riyadh 12631",
    "Al-Khobar Industrial Area, Al-Khobar 31952",
    "Tahlia Street, Al Olaya, Riyadh 11372",
    "King Abdul Aziz Road, Jeddah 23523",
    "PO Box 5000, Aramco, Dhahran 31311",

    # L — with country
    "King Fahd Road, Al Olaya, Riyadh 11564, Saudi Arabia",
    "Aramco Road, Dhahran, Eastern Province 31311, Saudi Arabia",
    "King Abdul Aziz Road, Jeddah 23523, Kingdom of Saudi Arabia",

    # XL multi-line
    "King Fahd Road\nAl Olaya\nRiyadh 11564\nSaudi Arabia",
    "Aramco Road\nDhahran, Eastern Province 31311\nKingdom of Saudi Arabia",

    # XXL
    "c/o Saudi Aramco\nAramco Road, Building 3\nDhahran, Eastern Province 31311\nKingdom of Saudi Arabia",

    # Tabular / ALL-CAPS
    "KING FAHD ROAD, AL OLAYA, RIYADH 11564, SAUDI ARABIA",

    # =========================================================================
    # UAE  (≥15 records)
    # Formats: free zone, district, emirate, PO Box common
    # =========================================================================

    # M
    "Downtown Dubai, Dubai",
    "Abu Dhabi, UAE",
    "DIFC, Dubai",

    # L
    "DIFC Gate Avenue, Dubai International Financial Centre, Dubai",
    "Business Bay, Al Abraj Street, Dubai 00000",
    "Jumeirah Lake Towers, Cluster T, Dubai",
    "Al Maryah Island, Abu Dhabi 44332",
    "Sheikh Zayed Road, Al Wasl, Dubai 10001",
    "PO Box 3500, Dubai, UAE",
    "PO Box 9000, Abu Dhabi, United Arab Emirates",
    "Plot M-12, Jebel Ali Free Zone, Dubai 17000",
    "Sheikh Mohammed Bin Rashid Blvd, Downtown Dubai",

    # L — with country
    "DIFC Gate Avenue, Dubai International Financial Centre, Dubai, UAE",
    "Al Maryah Island, Abu Dhabi 44332, United Arab Emirates",
    "Plot M-12, Jebel Ali Free Zone, Dubai 17000, UAE",

    # XL multi-line
    "DIFC Gate Avenue\nDubai International Financial Centre\nDubai\nUnited Arab Emirates",
    "Al Maryah Island\nAbu Dhabi 44332\nUnited Arab Emirates",
    "Sheikh Zayed Road, Al Wasl\nDubai\nUAE",

    # XXL
    "c/o Emirates National Oil Company Limited\nPO Box 27101\nDubai\nUnited Arab Emirates",

    # =========================================================================
    # ADDITIONAL EU COUNTRIES — diacritics and diverse formats
    # =========================================================================

    # Switzerland — German/French/Italian zones
    "Bahnhofstrasse 45, 8001 Zürich, Switzerland",
    "Rue du Rhône 14, 1204 Genève, Suisse",
    "Piazza Dante 1, 6900 Lugano, Svizzera",
    "Aeschenvorstadt 1, 4051 Basel, Switzerland",

    # Denmark
    "Gammeltorv 6, 1457 København K, Danmark",
    "Gammeltorv 6\n1457 København K\nDanmark",

    # Sweden
    "Sergels Torg 1, 111 57 Stockholm, Sverige",
    "Kungsgatan 12, 411 19 Göteborg, Sweden",

    # Poland — diacritics
    "ul. Marszałkowska 84/92, 00-514 Warszawa, Polska",
    "ul. Floriańska 1, 31-019 Kraków, Poland",
    "ul. Marszałkowska 84/92\n00-514 Warszawa\nPolska",

    # Turkey — diacritics, RTL-adjacent
    "Büyükdere Caddesi No. 127, 34394 Şişli, İstanbul, Türkiye",
    "Atatürk Bulvarı No. 3, 06100 Ulus, Ankara, Turkey",
    "Büyükdere Caddesi No. 127\nŞişli\nİstanbul\nTürkiye",

    # Belgium
    "Rue de la Loi 175, 1048 Bruxelles, Belgique",
    "Leuvenseweg 27, 1000 Brussel, België",

    # Portugal
    "Av. da Liberdade 36, 1250-147 Lisboa, Portugal",
    "Rua Augusta 24, 1100-053 Lisboa, Portugal",

    # Czech Republic
    "Wenceslas Square 1, 110 00 Praha 1, Česká republika",
    "Náměstí Svobody 17, 602 00 Brno, Czech Republic",

    # =========================================================================
    # ADDITIONAL REGIONS — misc format variety
    # =========================================================================

    # Canada
    "200 Bay Street, Royal Bank Plaza, Toronto, ON M5J 2J2",
    "1000 De La Gauchetière Street West, Montréal, QC H3B 4W5",
    "200 Bay Street, Royal Bank Plaza, Toronto, ON M5J 2J2, Canada",
    "200 Bay Street\nRoyal Bank Plaza\nToronto, ON M5J 2J2\nCanada",
    "c/o Royal Bank of Canada\n200 Bay Street, North Tower\nToronto, ON M5J 2J2\nCanada",

    # South Korea
    "26 Eulji-ro 5-gil, Jung-gu, Seoul 04539",
    "Samsung-ro 129, Suwon-si, Gyeonggi-do 16677",
    "26 Eulji-ro 5-gil, Jung-gu, Seoul 04539, South Korea",
    "26 Eulji-ro 5-gil\nJung-gu\nSeoul 04539\nRepublic of Korea",

    # Hong Kong
    "1 Harbour Road, Wan Chai, Hong Kong",
    "One IFC, 1 Harbour View Street, Central, Hong Kong",
    "1 Harbour Road, Wan Chai, Hong Kong SAR, China",
    "One IFC\n1 Harbour View Street\nCentral\nHong Kong",

    # Taiwan
    "No. 7, Section 5, Xinyi Road, Xinyi District, Taipei 110",
    "No. 7, Section 5, Xinyi Road\nXinyi District\nTaipei 110\nTaiwan",

    # Malaysia
    "Level 43, Menara Maxis, Kuala Lumpur City Centre, 50088 Kuala Lumpur",
    "Level 43, Menara Maxis, KLCC, 50088 Kuala Lumpur, Malaysia",
    "Level 43\nMenara Maxis, KLCC\n50088 Kuala Lumpur\nMalaysia",

    # Nigeria
    "Plot 1665A, Karimu Kotun Street, Victoria Island, Lagos 101241",
    "15A, Awolowo Road, Ikoyi, Lagos, Nigeria",
    "Plot 1665A, Karimu Kotun Street\nVictoria Island\nLagos 101241\nNigeria",

    # Egypt
    "1 Corniche El-Nil, Cairo Governorate, Egypt",
    "1 Corniche El-Nil\nCairo Governorate\nEgypt",

    # Russia
    "4 Shabolovka Street, Moscow 119049, Russia",
    "Liteyny Prospekt 4, Saint Petersburg 191028, Russia",
    "4 Shabolovka Street\nMoscow 119049\nRussian Federation",

    # =========================================================================
    # MIXED CASING VARIANTS
    # =========================================================================

    "1 martin place, sydney nsw 2000",
    "paseo de la reforma 222, cdmx 06600",
    "PLOT 42, SECTOR 5, GURUGRAM, 122001",
    "1600 PENNSYLVANIA AVE NW, WASHINGTON, DC 20500",
    "NO. 1 CENTURY AVENUE, PUDONG, SHANGHAI 200120",

    # =========================================================================
    # ADDITIONAL PO BOXES (various countries)
    # =========================================================================

    "PO Box 6753, Dubai, UAE",
    "PO Box 800, Riyadh 11421, Saudi Arabia",
    "PO Box 2020, Singapore 919802",
    "Postfach 10 20 30, 20095 Hamburg, Germany",
    "Boîte Postale 50001, 75020 Paris, France",

    # =========================================================================
    # FORMAT DIVERSITY — additional flavors
    # =========================================================================

    # Very long industrial / warehouse — XL
    "Warehouse 3, Gate 7, Jebel Ali Port, Dubai South, PO Box 17000, Dubai, UAE",
    "Shed 18, Nhava Sheva International Container Terminal, Navi Mumbai 400707, India",
    "Building 22, Jurong Island Chemical Complex, Singapore 627882",

    # Full multi-line XXL with company care-of lines
    "c/o DHL Global Forwarding\nBuilding 5, Dubai Logistics City\nDubai South, PO Box 17000\nDubai\nUnited Arab Emirates",
    "c/o Kuehne + Nagel\nPlot 14, Rajiv Gandhi Infotech Park, Phase 1\nHinjawadi, Pune 411057\nMaharashtra, India",
    "c/o CEVA Logistics\n200 Main Street, Sandton\nJohannesburg, Gauteng 2146\nSouth Africa",
    "c/o DP World, For Acct of ABC Trading Co.\nJebel Ali Free Zone Authority\nPO Box 17000, Dubai\nUnited Arab Emirates",
    "Attn: Mr. Li Wei, Import Manager\nNo. 1600 Jiuzhou Avenue, Bonded Zone\nZhuhai, Guangdong Province 519015\nP.R. China",
    "A/A: Responsable de Almacén\nBlvd. Puerto Aéreo 485, Módulo 4\nMoctezuma 2da Secc, Venustiano Carranza\nCiudad de México, CDMX 15620\nMéxico",
    "Attn: Cargo Control Dept.\nSingapore Customs\n55 Newton Road\nRevenue House\nSingapore 307987\nRepublic of Singapore",

    # Currency-locale specific non-ASCII (extra diacritics)
    "8 Curaçaostraat, 1077 GN Amsterdam, Netherlands",
    "Calle de la Cañada 12, 29620 Torremolinos, Málaga, Spain",
    "Rua Curaçao 45, Bela Vista, São Paulo, SP 01310-200, Brazil",
    "Vänortsvägen 3, 415 05 Göteborg, Sverige",

    # =========================================================================
    # S-BUCKET  (4–10 chars) — minimal city/code shorthands and PO Box
    # These appear in terse tabular fields; §3.2 requires ≥10 records.
    # =========================================================================
    "NYC, NY",
    "LA, CA",
    "SG 018",
    "Dubai",
    "Riyadh",
    "Mumbai",
    "Nairobi",
    "PB 1234",
    "Box 100",
    "Box 5500",
    "HK SAR",

    # =========================================================================
    # ADDITIONAL XXL ENTRIES  (121+ chars)
    # §3.2 requires ≥30 records in the XXL bucket for ADDRESS.
    # These are long multi-line addresses with full company care-of lines,
    # district, region, postal, and country.
    # =========================================================================

    # USA
    "c/o Cargill Incorporated, Receiving Department\n15407 McGinty Road West, Building 1\nWayzata, Minnesota 55391\nUnited States of America",
    "Attn: Freight Controller, Global Trade Division\n350 Fifth Avenue, 21st Floor, Suite 2100\nNew York, NY 10118\nUnited States of America",
    "For: Operations & Logistics Manager\n2200 Mission College Blvd, Mail Stop RN2-109\nSanta Clara, CA 95054\nUnited States",

    # UK
    "c/o Unilever PLC, Supply Chain Department\nUnilever House, 100 Victoria Embankment\nLondon EC4Y 0DY\nUnited Kingdom",
    "Attn: Import Compliance Team\nHSBC Holdings plc, 8 Canada Square\nCanary Wharf, London E14 5HQ\nUnited Kingdom",

    # Germany
    "z. Hd. Logistikleitung, Warenannahme Halle 3\nBASF SE, Carl-Bosch-Straße 38\n67056 Ludwigshafen am Rhein\nBundesrepublik Deutschland",
    "Herrn Lagerist, Eingangstor 7\nSiemens AG, Werner-von-Siemens-Straße 1\n80333 München\nDeutschland",

    # India
    "Attn: Warehouse Incharge, Gate No. 4\nReliance Industries Limited, Jamnagar Refinery Complex\nJamnagar, Gujarat 361142\nRepublic of India",
    "c/o Tata Steel Limited, Materials Management Department\nJamshedpur Works, Bistupur\nJamshedpur, Jharkhand 831001\nIndia",

    # China
    "Attn: Customs & Trade Compliance, Import Dock A\nHuawei Technologies Co., Ltd., Science & Technology Zone\nLonggang District, Shenzhen 518129\nPeople's Republic of China",
    "收货部门：物流管理处，装卸区3号\nNo. 500 Yishan Road, Xuhui District\nShanghai 200233\nP.R. China",

    # Japan
    "Attn: Procurement and Logistics Division, Goods Receipt Bay 2\nToyota Motor Corporation, 1 Toyota-cho\nToyota City, Aichi Prefecture 471-8571\nJapan",
    "宛先：入荷担当、輸入物流部\n三菱商事株式会社\n丸の内2丁目3番1号\n東京都千代田区 100-8086\n日本",

    # Singapore
    "c/o PSA International Pte Ltd, Container Terminal Operations\nPasir Panjang Terminal, 600A Pasir Panjang Road\nSingapore 118742\nRepublic of Singapore",

    # Australia
    "Attn: Import Operations, Receiving Bay 5\nBHP Billiton Limited, BHP Billiton Centre\n171 Collins Street, Level 18\nMelbourne, Victoria 3000\nAustralia",
    "c/o Toll Group, Freight Receipt Department\n380 St Kilda Road, Level 20\nMelbourne VIC 3004\nCommonwealth of Australia",

    # Brazil
    "A/C: Gerência de Logística e Importação, Portaria Principal\nPetróleo Brasileiro S.A., Avenida Henrique Valadares 28\nCentro, Rio de Janeiro, RJ 20231-030\nRepública Federativa do Brasil",
    "A/A: Coordenador de Operações Portuárias\nAv. Faria Lima 3500, Torre Sul, 12º Andar\nItaim Bibi, São Paulo, SP 04538-132\nBrasil",

    # Mexico
    "A/A: Gerencia de Logística Internacional, Módulo 7\nBlvd. Puerto Aéreo 485, Zona Industrial\nVenustiano Carranza, Ciudad de México, CDMX 15620\nEstados Unidos Mexicanos",

    # UAE
    "Attn: Port Operations Manager, Berth 14\nDP World Jebel Ali, Jebel Ali Free Zone\nPO Box 17000, Dubai\nUnited Arab Emirates",
    "c/o Emirates SkyCargo, Cargo Village, Zone C\nDubai World Central – Al Maktoum International Airport\nDubai South, PO Box 23000\nDubai, United Arab Emirates",

    # South Africa
    "Attn: Import / Export Manager, Dock 22\nDurban Container Terminal, Island View\nDurban, KwaZulu-Natal 4004\nRepublic of South Africa",

    # Saudi Arabia
    "Attn: Procurement Division, Goods Receiving Station 12\nSaudi Aramco, Dhahran Administrative Area\nDhahran, Eastern Province 31311\nKingdom of Saudi Arabia",

    # Multi-country transit address
    "c/o Maersk Line\nPort of Rotterdam, Maasvlakte 2, Rotterdam Shortsea Terminal\nMaasvlakteplein 1, 3199 LE Rotterdam\nThe Netherlands",
    "c/o CMA CGM Group, Container Logistics Platform\nPort of Hamburg, Terminal Burchardkai\nOstbahnhof 4, 20457 Hamburg\nFederal Republic of Germany",

    # =========================================================================
    # ADDITIONAL XL ENTRIES  (61–120 chars) — filling the ≥70 minimum
    # =========================================================================

    # Single-line XL — US / UK / EU with full detail
    "2200 Mission College Blvd, Building C, Suite 100, Santa Clara, CA 95054, USA",
    "410 Terry Ave N, Building D, Floor 3, Seattle, WA 98109, United States",
    "30 St Mary Axe, 10th Floor, London EC3A 8BF, United Kingdom",
    "Bockenheimer Landstraße 24, Deutsche Bank Tower, 60323 Frankfurt, Germany",

    # =========================================================================
    # ADDITIONAL XXL ENTRIES  (121+ chars) — filling the ≥30 minimum
    # =========================================================================

    # UK
    "Attn: Logistics and Supply Chain, Goods Inbound Bay 9\nUnilever PLC, Port Sunlight\nMerseyside CH62 4UY\nUnited Kingdom",

    # France
    "À l'attention du Responsable Logistique, Entrepôt B\nTotal Énergies SE, 2 Place Jean Millier\nLa Défense, 92400 Courbevoie\nFrance",
    "A/A Service Import-Export, Bâtiment C, Quai de Déchargement\n15 Rue de Rivoli, 7ème Étage\n75001 Paris\nFrance",

    # Netherlands
    "t.a.v. Importafdeling, Ontvangstdok 5\nShell Nederland B.V., Hofplein 20\n3032 AC Rotterdam\nNederland",

    # Spain
    "A/A: Departamento de Compras Internacionales, Nave 14\nRepsol S.A., Calle Méndez Álvaro 44\n28045 Madrid\nEspaña",

    # India
    "Attn: Store Keeper, Material Receipt Section, Plant B\nInfosys Limited, Electronics City Phase 2\nBengaluru, Karnataka 560100\nRepublic of India",
    "c/o Hindustan Unilever Limited, Supply Chain Division\nTechniplex Complex, Off Veer Savarkar Flyover\nGoregaon West, Mumbai 400062\nMaharashtra, India",

    # China
    "Attn: Import Coordination, Bonded Warehouse, Zone A, Unit 12\nNo. 1 Century Avenue, Pudong New Area\nShanghai 200120\nPeople's Republic of China",
    "c/o COSCO Shipping, Container Reception, Terminal 2, Berth 18\nNo. 1 Renmin Road, Pudong New Area\nShanghai 200135\nP.R. China",

    # Japan
    "Attn: Logistics Division, Receiving Section, Dock 4\nSumitomo Corporation, 1-8-11 Harumi, Chuo-ku\nTokyo 104-8610\nJapan",

    # Singapore
    "Attn: Customs & Trade Compliance Manager, Warehouse Zone C, Bay 12\nSingapore Logistics Park, 5 Benoi Sector\nSingapore 629845\nRepublic of Singapore",

    # Australia
    "c/o Rio Tinto Limited, Shipping and Freight Division\n120 Collins Street, Level 33\nMelbourne, Victoria 3000\nCommonwealth of Australia",

    # Brazil
    "c/o Vale S.A., Coordenação de Logística e Transporte Marítimo\nPraia de Botafogo 186, Torre A, 18° Andar\nBotafogo, Rio de Janeiro, RJ 22250-145\nBrasil",

    # South Korea
    "Attn: Import Clearance Team, Cargo Reception Dock 7\nSamsung C&T Corporation, Samsung-ro 129\nSuwon-si, Gyeonggi-do 16677\nRepublic of Korea",

    # Canada
    "Attn: Supply Chain Manager, Receiving Dock B\nSuncor Energy Inc., 150 6th Avenue SW\nCalgary, Alberta T2P 3Y7\nCanada",

    # Mexico
    "A/A: Jefe de Almacén, Zona de Recibo, Puerta Norte\nPetróleos Mexicanos, Av. Marina Nacional 329\nCol. Verónica Anzures, Miguel Hidalgo\nCiudad de México, CDMX 11300\nEstados Unidos Mexicanos",

    # Nigeria
    "Attn: Operations Manager, Bonded Warehouse, Zone B\nNigerian National Petroleum Corporation Limited\nCNPC Towers, Herbert Macaulay Way\nAbuja FCT 900211\nFederal Republic of Nigeria",

    # South Africa — extra
    "c/o Sasol Limited, Supply Chain and Procurement Division\n1 Sturdee Avenue, Rosebank\nJohannesburg, Gauteng 2196\nRepublic of South Africa",

    # Russia
    "Attn: Otdel Logistiki, Sklad 7, Vorota 3\nPublichnoe Aktsionernoe Obshchestvo Gazprom\n16 Nametkina Street, Cheremushki District\nMoscow 117420\nRussian Federation",

]
