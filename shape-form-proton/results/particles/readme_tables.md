Report-mérce (teszt-NLL, kisebb = jobb; zárójelben a paraméterszám; Δ = páros különbség a × Ā-val újraillesztett baseline-hoz):

| részecske | baseline (report) | baseline × Ā | új, n = 2 | új (1-SE) | Δ új − baseline × Ā | MLP report / finomhangolt | rés bezárva |
|---|---:|---:|---:|---:|---:|---:|---:|
| elektron | 14,8479 (33) | 14,8406 | 14,8173 (53) | **14,8167** (75) | -0,0239 ± 0,0009 | 14,8068 / 14,8067 (2785) | 70 % |
| pion | 18,1563 (27) | 18,1420 | 18,1239 (87) | **18,1235** (101) | -0,0186 ± 0,0007 | 18,1221 / 18,1208 (3841) | 88 % |
| proton | 18,2536 (45) | 18,2331 | 18,1952 (87) | **18,1940** (143) | -0,0391 ± 0,0016 | 18,1922 / 18,1898 (881) | 90 % |
| müon | 16,0899 (59) | 16,0846 | 16,0132 (81) | **16,0120** (129) | -0,0726 ± 0,0011 | 16,0046 / 16,0034 (2785) | 89 % |

3D-mérce (a detektor-szeletre normálva; minden modell így tanítva):

| részecske | baseline | új, n = 2 | új (1-SE) | Δ új − baseline | MLP | rés bezárva |
|---|---:|---:|---:|---:|---:|---:|
| elektron | 14,8279 | 14,8076 | **14,8072** (64) | -0,0207 ± 0,0008 | 14,7976 | 68 % |
| pion | 18,1170 | 18,1015 | **18,1014** (101) | -0,0157 ± 0,0007 | 18,0920 | 63 % |
| proton | 18,2003 | 18,1676 | **18,1660** (143) | -0,0343 ± 0,0015 | 18,1614 | 88 % |
| müon | 16,0832 | 16,0163 | **16,0144** (129) | -0,0688 ± 0,0011 | 16,0000 | 83 % |

Független második minta (s2, más véletlenmag; Δ = új − baseline, report-mérce / 3D):

| részecske | események | Δ report-mérce | Δ 3D | rés bezárva |
|---|---:|---:|---:|---:|
| elektron | 99997 | -0,0236 ± 0,0003 | -0,0199 ± 0,0003 | 73 % / 71 % |
| pion | 99714 | -0,0212 ± 0,0004 | -0,0181 ± 0,0003 | 90 % / 69 % |
| proton | 99948 | -0,0409 ± 0,0007 | -0,0363 ± 0,0006 | 91 % / 89 % |
| müon | 99783 | -0,0739 ± 0,0007 | -0,0699 ± 0,0006 | 90 % / 84 % |

A kiegészítő tagok hozzájárulása (végleges formula − alapváltozat, teszt, páros):

| részecske | változat | report-mérce | 3D |
|---|---|---:|---:|
| elektron | plateau+all+curv | -0,0091 ± 0,0014 | -0,0086 ± 0,0014 |
| pion | plateau_track+all+all4 | -0,0013 ± 0,0003 | -0,0000 ± 0,0003 |
| proton | plateau_track+all+all4 | -0,0023 ± 0,0004 | -0,0020 ± 0,0004 |
| müon | full_att+all+all4 | -0,0080 ± 0,0006 | -0,0078 ± 0,0005 |

Változatválasztás (validációs NLL, report-mérce × Ā, n = 2; a választott a legkevesebb paraméteres a legjobbtól 1 SE-n belül):

| részecske | választott | zápor + nyom + csúcs | zápor + nyom (csillapítással) + csúcs | zápor + csúcs | plató-zápor + csúcs | plató-zápor + nyom + csúcs | fokszám (report / 3D) |
|---|---|---:|---:|---:|---:|---:|---:|
| elektron | plató-zápor + csúcs + all + curv | 14,8084 | 14,8085 | 14,8096 | 14,8069 | 14,8064 | 4 / 3 |
| pion | plató-zápor + nyom + csúcs + all + all4 | 18,1274 | 18,1266 | 18,1290 | 18,1281 | 18,1239 | 3 / 3 |
| proton | plató-zápor + nyom + csúcs + all + all4 | 18,2093 | 18,2052 | 18,2202 | 18,2202 | 18,2043 | 6 / 6 |
| müon | zápor + nyom (csillapítással) + csúcs + all + all4 | 16,0359 | 16,0216 | 16,0435 | 16,0270 | 16,0255 | 6 / 6 |

Burok-testek (95 % energia a záporok 98 %-ában; átlag a 20 binre, liter; zárójelben a teszt-lefedés):

| részecske | csepp (report 03, újraillesztve): tartomány / szelet | szuper-tojás: tartomány / szelet | különbség a szeletben |
|---|---:|---:|---:|
| elektron | 25,9 / 24,4 (0,977) | 25,9 / 24,2 (0,977) | -1.1 % |
| pion | 1478 / 1335 (0,977) | 1423 / 1302 (0,976) | -2.5 % |
| proton | 1600 / 1428 (0,977) | 1544 / 1395 (0,976) | -2.3 % |

Ellenőrzés (verify.json):
- electron: report-mérce újraszámolva max |eltérés| 1.7e-05; 3D nsub 8 vs 2: -2.6e-04; konzisztens: True; normáltság 3.6e-07; választás tesztet nem használ: True; teszt és s2 előjele egyezik: True
- pion: report-mérce újraszámolva max |eltérés| 3.1e-05; 3D nsub 8 vs 2: -7.2e-05; konzisztens: True; normáltság 3.0e-07; választás tesztet nem használ: True; teszt és s2 előjele egyezik: True
- proton: report-mérce újraszámolva max |eltérés| 2.9e-05; 3D nsub 8 vs 2: -1.3e-04; konzisztens: True; normáltság 3.6e-07; választás tesztet nem használ: True; teszt és s2 előjele egyezik: True
- muon: report-mérce újraszámolva max |eltérés| 1.1e-05; 3D nsub 8 vs 2: -1.4e-04; konzisztens: True; normáltság 7.2e-07; választás tesztet nem használ: True; teszt és s2 előjele egyezik: True
