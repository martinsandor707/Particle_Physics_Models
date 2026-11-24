# Záporalak – jobb 3D alak (elektron, pion, proton, müon)

A `lib/report_hu.pdf` (Sofiya `shape-form` notebookjai) folytatása: ugyanaz az adat, felosztás (70/15/15 %, `SEED = 1`), rács és
eseményenkénti NLL; a közös kód (`00_common.ipynb`) változatlan. Az adat az ePIC LFHCal önálló DD4hep-szimulációja (60 réteg × [4 mm
szcintillátor + 15,2 mm acél + 1,3 mm Kapton/levegő], előtte 40,4 mm acél; a rétegszerkezetre X₀ = 23,5 mm, λ_I = 212 mm, λ_π = 260 mm,
R_M = 21,9 mm; forrás: eic/epic `compact/hcal/lfhcal.xml`).

## Röviden

- **A formula.** Fizikailag külön tagok összege (zápor + nyom + belépési csúcs + hatótáv-végi folt + távoli udvar), 75–143 paraméter,
  közvetlenül a szimulált találatokra illesztve. A régi képlet és a háló közti rés **70–90 %-át** zárja be; egy független második
  mintán (s2) ugyanez jön ki.
- **Mennyire írja le a záport.** Az átlagos profil sugárirányban elektronnál és protonnál a hálóhoz közeli, müonnál jobb nála; mélységben
  a háló jobb, de hadronoknál (p ≥ 2 GeV) a formula eltérése a profil tömbjében csak ~1,5–2 %.
- **Fizika.** A fizikailag értelmes paraméterek többnyire a detektor anyagából várt értékek közelében vannak (müon többszörös szórása,
  hatótávok, csillapítási hosszak). A szimulációs igazság szerint az elektron távoli udvara azonnali elektromágneses energia, a hadronok
  távoli energiájának fele–háromnegyede neutronoktól jön.
- **Alakok.** A legjobb 98 %-os burok a szuper-tojás (proton: 3,5 %-kal kisebb az újraillesztett cseppnél); a kalibrált izoszint nem kisebb.
- **Szögfüggő kiegészítés (nem modellpredikció, lásd 6.).** Ha a beesési szöget bemenetként megadjuk, és néhány belépési paramétert a
  szöggel együtt illesztünk, a 3D-likelihood tovább javul (müonnál −0,045 nat).
- **Ami hiányzik.** A záporonkénti ingadozás (a formula átlag), a belépés pontos szögfüggése, és a 100 ns után érkező energia
  (hadronoknál legalább 3–5 %).

## Részecskénként – mi jött ki, mi következik belőle

(A fogalmak magyarázata az 1. fejezetben, a számok forrása a 2–7. fejezet.)

**Elektron** – zápor + belépési csúcs + távoli udvar, nyom nincs (64 paraméter).
- Kis energián az első réteg sok energiát kap: a belépési csúcs 1 GeV-en az energia 17 %-a, 20 GeV-en már csak 0,35 %; a többi a zápor.
- A zápor magja σ ≈ 16–19 mm, kb. 0,8 Molière-sugár: a keresztmetszet a várt elektromágneses záporé.
- A zápor maximuma 35 → 76 mm (1 → 20 GeV), a PDG-becslés 23 → 94 mm: az irány jó, de a formula energiafüggése gyengébb a vártnál.
- A távoli udvar (az energia ~1 %-a, mélységi hossza ~0,77 m) a szimuláció szerint azonnali e±/γ energia, nem neutron; a formula
  5–20 GeV-en ~30 %-kal alulbecsli.
- Itt maradt a legtöbb a hálóhoz képest (rés 70 %): a mély farok (0,35–0,44 m-en 6–11 %-kal kevesebb) és a cellánál keskenyebb mag.
- A szög alig számít (−0,002 nat); az ingadozást 3 főkomponens ~73 %-ban leírja; a legjobb burok kb. akkora, mint a csepp (26 L).
- **Megállapítás:** az elektronzáport a klasszikus EM-alak jól leírja, a hiány a részletekben (maximum helye, mély farok) van.

**Müon** – kis zápor + nyom, minden kiegészítéssel (129 paraméter); az energia 87–93 %-a a nyomban.
- Itt a legnagyobb a nyereség (−0,073 nat a régi képlethez, rés 89 %): a régi képletben nem volt nyom.
- A nyom szélesedése többszörös szórásból jön: k ≈ 1,41 egyezik a Highland-becsléssel (1,35–1,41), c ≈ 74 mm 17 %-kal nagyobb a vártnál.
- A Bragg-felfutás kitevője 0,42 (elmélet 0,43), a hatótáv ≈ 1,06 m/GeV · T, egyezik a CSDA-táblával; T ≳ 1,3 GeV felett a müon
  átmegy a detektoron.
- A megálló μ⁻ a vasban befogódik (neutronok, gammák), ezért kis impulzuson a távoli energia akár 20 %-a neutronból jön.
- Sugárirányban jobb a hálónál, mélységben rosszabb: a legmélyebb ~13 cm-en az adat leesik (a nyomok egy része a peremen távozik),
  a formula nem.
- A szögfüggő kiegészítés itt hozza a legtöbbet (−0,045 nat), de csak úgy, hogy a tagok szerepet cserélnek, ezért fizikailag nem értelmezhető.
- Legjobb burok: szuper-tojás vagy kapszula (~38 L), 19 %-kal kisebb a cseppnél.
- **Megállapítás:** a müon „alakja” valójában egy fizikailag jól leírható nyom; az illesztett paraméterek tankönyvi értékeket adnak vissza.

**Proton** – zápor + nyom, minden kiegészítéssel (143 paraméter).
- Két rezsim: 1 GeV-en a proton főleg megáll (nyom 61 %, belépési csúcs 24 %, zápor 15 %), 5–20 GeV-en záport kelt (zápor 74–81 %).
- A nyereség (−0,039 nat, rés 90 %) zöme a megálló protonoké; ami a hálóhoz képest maradt, annak ~60 %-a is p < 2 GeV-en van.
- A hatótáv T = 0,5 / 1 GeV-en 184 / 721 mm (CSDA: 219 / 694 mm); a csillapítási hossz 266 mm ≈ 1,26 λ_I.
- 2 GeV felett a „nyom” már effektív: több energiát visz (17–27 %), mint a proton saját ionizációja (5–12 %), mert a hadronmag egy
  részét is leírja.
- A távoli energia fele–háromnegyede neutronoktól jön; ezt a zápor exponenciális udvara viszi. A később (≥ 5 ns) elsőként talált
  cellák az energia legalább 8–11 %-át adják, és 2,5–3-szor szélesebben vannak: két (gyors + lassú) udvar-tag lenne a következő lépés.
- A profil tömbjében az eltérés 2,1 %; sugárirányban a hálóhoz közeli.
- Legjobb burok: szuper-tojás, 1544 L, 3,5 %-kal kisebb a cseppnél.
- **Megállapítás:** a megálló proton fizikája (hatótáv, csillapítás) jól megfogható a formulában, a nagy energiás hadronzápor átlaga
  is; az eseményenkénti ingadozása viszont nem (3 főkomponens csak a variancia 40–66 %-a).

**Pion** – mint a proton (101 paraméter); a nyom 26–53 %.
- Rés 88 %; a kiegészítő tagok itt alig adnak (~0,001 nat), a lényeg a zápor + nyom.
- A csillapítási hossz 339 mm ≈ 1,31 λ_π; a pion/proton arány 1,28 (várt 1,23): a pion mélyebbre jut a protonnál, ahogy kell.
- A profil tömbjében itt a legkisebb az eltérés (1,5 %).
- A távoli energia 45–60 %-a neutronból jön; a később talált cellák az energia legalább 6–8 %-át adják.
- Legjobb burok: szuper-tojás, 1423 L, 3,8 %-kal kisebb a cseppnél.
- A pion 3D-hálója degenerált, ezért pionnál a 3D-összevetés a hálóval nem megbízható (a report-mércés igen).
- **Megállapítás:** a pion a protonhoz hasonlóan viselkedik, hosszabb csillapítási hosszal; az átlagos alak jó, az ingadozás sokdimenziós.

## Fizikai értelmezés

Mit jelentenek ezek fizikailag? A ✓-lal jelölt pontok számolással ellenőrzöttek; a többi értelmezés: összhangban van az adattal, de
külön mérés nem készült rá.

- **A Bragg-kitevő a hatótáv-törvényből következik (✓).** Ha a hatótáv R ∝ T^a, akkor a fajlagos energialeadás a pálya végétől mért
  távolsággal dE/dx ∝ (R − z')^(−(1 − 1/a)). Bragg–Kleeman a = 1,77-tel ez 0,43; a müonra illesztett q = 0,42. A formula tehát a
  megálló müonnál a valódi Bragg-felfutást tanulta meg.
- **A müon hatótáva lineáris T-ben (✓),** mert a vizsgált tartományban (T ≈ 0,15–1,3 GeV, βγ ≈ 0,6–13) a müon a minimálisan ionizáló
  tartomány közelében van, ahol dE/dx közel állandó. Ezért R ≈ T / (dE/dx) – az illesztett 1,06 m/GeV a CSDA-táblával egyezik.
- **A többszörös szórás kitevője az előlap miatt kisebb 1,5-nél (✓).** Vékony anyagban a nyom oldalirányú eltérése ∝ z'^1,5
  (Highland). A 40 mm-es acél előlap már a detektor előtt szögszórást ad, és ez négyzetesen hozzáadódik: az effektív kitevő 1,35–1,41
  lesz. A c 17 %-kal nagyobb a vártnál; ez valószínűleg a pálya menti energiaveszteség (a müon lassul, és a szórás ∝ 1/p nő), amit a
  becslés nem tartalmaz.
- **A proton illesztett hatótáva kis energián rövidebb a CSDA-nál (−16 %, T = 0,5 GeV).** A CSDA azt feltételezi, hogy a proton
  kölcsönhatás nélkül fékeződik le. A 219 mm-es pálya azonban összemérhető a magkölcsönhatási hosszal (212 mm): a protonok kb.
  55–65 %-a a pálya vége előtt magkölcsönhatásba lép, ezért az átlagos „nyom” hamarabb véget ér.
- **A csillapítási hossz ~1,3-szorosa a magkölcsönhatási hossznak.** A λ_att nem azt méri, mikor szűnik meg a primer, hanem azt, hogy
  mikor fogy el a tengely menti, nyomszerű energia. Rugalmas és kvázirugalmas ütközés után, illetve ha egy vezető részecske viszi
  tovább az energia nagy részét, a tengely menti lerakódás folytatódik. Ezért természetes, hogy λ_att > λ_I. A pion/proton arány
  (1,28, várt 1,23) a kisebb pion–mag hatáskeresztmetszetet követi.
- **Két rezsim a protonnál.** p = 1 GeV/c-nél (T = 0,43 GeV) a CSDA-hatótáv 165 mm, ebből ~51 mm az előlapban fogy el. A proton
  tehát az első ~5 rétegben megáll, ezért a belépési csúcs és a nyom viszi az energiát. 5 GeV felett a magkölcsönhatás dominál, és
  hadronzápor lesz.
- **Neutronok és a lassú, széles udvar.** A neutronok töltés nélküliek: messzire jutnak, és csak közvetve (meglökött protonok a
  hidrogéndús szcintillátorban, befogási gammák) adnak le energiát. A MeV-es neutronok lelassulása és befogása ns–µs-os folyamat. Ez
  megmagyarázza, miért szélesebb és mélyebb a később talált cellák energiája, és miért kell egy gyors és egy lassú udvar-tag.
- **Protonnál több a neutron, mint pionnál** (távoli energia 51–75 % kontra 45–60 %; késleltetett 8–11 % kontra 6–8 %). Ez
  összhangban van az ismert barionszám-hatással: a proton záporában a vezető barion miatt kevesebb π⁰ keletkezik, így kisebb az
  elektromágneses és nagyobb a hadronos/neutronos rész.
- **Miért sokdimenziós a hadronzápor ingadozása?** Egy EM-zápor eseményenként főleg a kezdőpontjában és a hosszában különbözik (1–2
  szabadsági fok, 3 főkomponens ~3/4). Egy hadronzápornál eseményenként változik:
  - az első kölcsönhatás helye;
  - a π⁰-ok (tehát az EM-rész) aránya és helye;
  - a neutronok száma;
  - a magkötési energia, ami láthatatlanul elvész.

  Ezek nagyrészt függetlenek egymástól, ezért 3 főkomponens a varianciának csak 40–50 %-át írja le.
- **Elektron: a belépési csúcs és a zápormaximum.** Az előlap 2,3 X₀ vastag, így a zápor már a detektor előtt elindul. Kis energián a
  maximum közel van a belépéshez, és ennek a korai résznek a nagy részét a csúcs-tag viszi (1 GeV-en 17 %). Valószínűleg ezért
  csúszik az illesztett maximum kis energián későbbre, és ezért gyengébb az energiafüggése (35 → 76 mm a várt 23 → 94 mm helyett). A
  20,5 mm-es rétegosztás is korlátozza, mennyire élesen látszik a maximum.
- **Elektron: a távoli udvar** azonnali e±/γ energia. Valószínűleg néhány MeV-es fotonokról van szó: ezek a legkevésbé elnyelt
  fotonok (a Compton-tartomány elnyelési minimuma közelében), ezért jutnak messzire.

Tartalom: Részecskénként · Fizikai értelmezés · 1. Fogalmak · 2. A formula · 3. Pontosság · 4. Alakok · 5. Ellenőrző mérések · 6. Szögfüggő kiegészítés · 7. Továbbiak
(időbeli szétválasztás, ingadozás) · 8. Megállapítások · 9. Korlátok · 10. Fájlok.

## 1. Fogalmak – mit jelentenek a számok

| fogalom | jelentés |
|---|---|
| z', r | mélység a beesés irányában a zápor kezdetétől (az első találati rétegtől), és a tengelytől mért távolság [mm] |
| p, T | a beeső részecske impulzusa [GeV/c] és mozgási energiája T = √(p² + m²) − m [GeV] |
| NLL [nat] | eseményenként a lerakódások átlagos negatív log-valószínűsége a modell szerint; **kisebb = jobb**. Csak ugyanazzal a mércével tanított modellek NLL-je hasonlítható össze; a különbségek a fontosak, nem az abszolút érték |
| Δ (páros) | két modell NLL-jének különbsége ugyanazokon az eseményeken, ± a standard hibája; negatív = az első jobb |
| rés bezárva | (régi képlet − új formula) / (régi képlet − háló): a régi képlet és a háló közti távolság hány %-át hozza be a formula |
| report-mérce, × Ā | a report likelihoodja; a modell egy szögekre átlagolt detektor-elfogadási sablonnal (Ā) szorozva |
| 3D-mérce | a likelihood eseményenként a valódi detektor-szeletre (a ferde belépési sík és a hátlap közé) normálva |
| teszt, s2 | a teszthalmaz (15 %), illetve egy független második szimulációs minta (~100 000 esemény), amely semmilyen döntésben nem szerepelt |
| n / paraméter | a polinomok fokszáma és a formula szabad paramétereinek száma |
| S (szeparációs erő) | két normált hisztogram eltérése (CaloChallenge): S = ½ Σ (h − g)² / (h + g); 0 = azonos, 1 = nem fednek át. Kis eltérésnél √(4S) ≈ az energiával súlyozott négyzetes relatív eltérés; a táblázatokban × 10⁻³ egységben |
| padló | az adat két véletlen felének S-e: ekkora eltérés tökéletes modellnél is a véges minta miatt |
| „a detektorban” arány | egy tag részesedése a detektorba (a belépési sík mögé) eső energiából |

## 2. A formula

**Miből illesztve.** A paraméterek közvetlenül a tanítóhalmaz találataira vannak illesztve (energiával súlyozva, maximum likelihood),
nem háló-kimenetre. A tagok és a fokszám a validáción lettek kiválasztva. A háló csak mérce.

`ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H`

Minden tag külön 1-re normált eloszlás; a w súlyok a tagok energiarészesedései (összegük 1). σ(·) a logisztikus „lépcső”.

| tag | képlet | jelentés |
|---|---|---|
| zápor S | `L(z')·T(r│z')` | a hadron- vagy EM-zápor |
| – mélységprofil | plató: `σ((z'−z_s)/w1)·σ((ℓ−z')/w2)·e^(κz')`; müon: gamma `s^(α−1)e^(−(α−1)s/t_max)` | z_s: kezdet, w1: felfutás, ℓ: vég, w2: lecsengés, κ: lassú meredekség |
| – keresztmetszet | `f1·Gauss(r;σ) + f2·R²/(π(r²+R²)²) + f3·e^(−r/λ)/(2πλ(r+1 mm))` | keskeny mag (σ), Cauchy-test (R), exponenciális udvar (λ); f1–f3 arányok; mind a mélységgel változik |
| nyom U | `σ((z'−z_t0)/4 mm)·σ((R(T)−z')/w_R)·e^(−z'/λ_att)·(1−z'/R+0,03)^(−q)` × keresztmetszet | a primer részecske nyoma: R(T) = R1·T^k_R − R0 a hatótáv, λ_att a csillapítás (első magkölcsönhatás), q a Bragg-felfutás |
| – a nyom szélessége | `σ(z')² = σ0² + (c·(z'/1 m)^k / βp)²`, plusz `f·e^(−r/l)` udvar | többszörös szórás miatti szélesedés (c, k), delta-elektron-udvar (f, l) |
| csúcs D | Gauss a belépésnél | az első réteg energiája |
| folt E | Gauss a z' = R(T) helyen | megálló részecske vége (Bragg-csúcs, befogás) |
| távoli udvar H | `σ(z'/10 mm)·e^(−z'/L)/(r+1 cm)` | diffúz, messzire jutó energia |

Minden paraméter az `x = (ln T − v0)/vs` polinomja (a legtöbb n-edfokú, néhány állandó). Részecskénként a teljes leírás, minden illesztett
érték és együttható: `results/particles/<részecske>/formula.md`.

**Illesztett értékek** (3D-mérce):

| részecske | tagok | n / param. | a detektorba eső energia | fontos paraméterek (zárójelben a detektor anyagából várt érték) |
|---|---|---|---|---|
| elektron | plató-zápor + csúcs + távoli udvar | 3 / 64 | zápor 82 % (1 GeV) → 99 % (20 GeV); csúcs 17 % → 0,35 %; távoli udvar 0,7–1,0 % | mag σ ≈ 16–19 mm (R_M = 21,9 mm); zápormaximum 35 → 76 mm (PDG: 23 → 94 mm) |
| müon | gamma-zápor + nyom + minden kiegészítés | 6 / 129 | nyom 87–93 % | szélesedés c ≈ 74 mm, k ≈ 1,41 (Highland: 62–63 mm, 1,35–1,41); q ≈ 0,42 (0,43); hatótáv ≈ 1,06 m/GeV · T (CSDA ≈ 1,06–1,07) |
| proton | plató-zápor + nyom + minden kiegészítés | 6 / 143 | 1 GeV: nyom 61 %, csúcs 24 %, zápor 15 %; 5–20 GeV: zápor 74–81 %, nyom 17–21 % | hatótáv T = 0,5 / 1 GeV-en 184 / 721 mm (CSDA: 219 / 694 mm); λ_att ≈ 266 mm (λ_I = 212 mm) |
| pion | mint a proton | 3 / 101 | nyom 26–53 % | λ_att ≈ 339 mm (λ_π = 260 mm) |

Hadronoknál a nyom szélesedése, delta-udvara és Bragg-kitevője effektív: a hadronmag egy részét is ezek írják le.

<!-- KEPLETEK:START (scripts/readme_particle_formulas.py generálja) -->

### Részecskénkénti képletek (3D-mérce, a beírt számok az illesztett értékek)

A fenti általános képlet részecskénként: melyik tag van benne, az állandó paraméterek behelyettesítve, a p-függő paraméterek 1, 5 és 20 GeV/c-nél. Az „energiarész” a detektorba (a ferde belépési sík mögé) eső energia megoszlása. A polinom-együtthatók: `results/particles/<részecske>/formula_coefficients_slab.csv`; minden impulzus és a report-mérce: `formula.md`. Ha a hatótáv nagyobb a detektornál (~1,3 m), a részecske átmegy rajta.

**Elektron** (m = 0,511 MeV; polinomfok és paraméterszám: n = 3, 64)

`ρ(z', r | p) = w_zápor·S + w_csúcs·D + w_távoli·H`

- Zápor mélységprofilja plató: `L(z') ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ·z'/1 m)` (értékek lent).
- Zápor keresztmetszete: `T(r│z') = f_mag·Gauss(r; σ) + f_test·Cauchy(r; R) + f_udvar·e^(−r/λ)/(2πλ(r + 1 mm))`; σ, R, λ és az arányok a mélységgel változnak (lent z' = 100 mm-en).
- Csúcs a belépésnél: Gauss (hely és szélességek lent). Távoli udvar: `H ∝ σ(z'/10 mm)·e^(−z'/767 mm)/(r + 1 cm)`.

| | p = 1 GeV/c | p = 5 GeV/c | p = 20 GeV/c |
|---|---:|---:|---:|
| T (mozgási energia) | 0,999 GeV | 5 GeV | 20 GeV |
| energiarész: zápor | 82,4 % | 94,7 % | 98,7 % |
| energiarész: csúcs | 17 % | 4,38 % | 0,351 % |
| energiarész: távoli udvar | 0,666 % | 0,899 % | 0,951 % |
| zápor kezdete z_s / felfutás w1 | 5,51 / 19,9 mm | 15,1 / 24 mm | 25,2 / 26,3 mm |
| zápor vége ℓ / lecsengés w2 | 42,8 / 60,2 mm | 82,3 / 65,5 mm | 154 / 66,2 mm |
| lassú meredekség κ | −1,44 /m | −1,16 /m | −1,33 /m |
| mag σ / Cauchy R / udvar λ (z' = 100 mm) | 19 / 34,8 / 39,5 mm | 16,5 / 26,7 / 39,5 mm | 15,8 / 20,7 / 39,5 mm |
| arány mag / test / udvar (z' = 100 mm) | 0,74 / 0,13 / 0,13 | 0,82 / 0,04 / 0,13 | 0,85 / 0,02 / 0,13 |
| csúcs helye / σ_z / σ_r | −8,84 / 21,9 / 8,88 mm | −10,6 / 24,6 / 9,15 mm | −12,6 / 18,3 / 9,74 mm |

**Müon** (m = 105,7 MeV; polinomfok és paraméterszám: n = 6, 129)

`ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H`

- Zápor mélységprofilja gamma: `L(z') ∝ s^(α−1)·e^(−(α−1)s/t_max)`, s = z' − z0 (értékek lent).
- Zápor keresztmetszete: `T(r│z') = f_mag·Gauss(r; σ) + f_test·Cauchy(r; R) + f_udvar·e^(−r/λ)/(2πλ(r + 1 mm))`; σ, R, λ és az arányok a mélységgel változnak (lent z' = 100 mm-en).
- Nyom: hatótáv `R(T) = 1057 mm · (T/GeV)^1,02 − 85 mm`; csillapítás `e^(−z'/1211 mm)`; Bragg-felfutás `(1 − z'/R + 0,03)^(−0,42)`; szélesség `σ(z')² = (15 mm)² + (74,4 mm · (z'/1 m)^1,41 / βp)²`; delta-elektron-udvar `e^(−r/51,4 mm)` (aránya lent).
- Folt a hatótáv végén: Gauss, σ_z = 73,1 mm, σ_r = 85,6 mm.
- Csúcs a belépésnél: Gauss (hely és szélességek lent). Távoli udvar: `H ∝ σ(z'/10 mm)·e^(−z'/1309 mm)/(r + 1 cm)`.

| | p = 1 GeV/c | p = 5 GeV/c | p = 20 GeV/c |
|---|---:|---:|---:|
| T (mozgási energia) | 0,9 GeV | 4,9 GeV | 19,9 GeV |
| energiarész: zápor | 2,42 % | 6,11 % | 1,39 % |
| energiarész: nyom | 90,9 % | 88 % | 92,5 % |
| energiarész: csúcs | 6,38 % | 5,79 % | 5,97 % |
| energiarész: folt | 0,239 % | < 0,01 % | < 0,01 % |
| energiarész: távoli udvar | 0,0852 % | 0,093 % | 0,0942 % |
| zápor kezdete z0 | −48,6 mm | −57,1 mm | −71 mm |
| zápor maximuma t_max | 5,36 mm | 90,9 mm | 240 mm |
| alak α − 1 | 0,0747 | 0,00916 | 0,0508 |
| mag σ / Cauchy R / udvar λ (z' = 100 mm) | 11,2 / 9,51 / 52,3 mm | 11,7 / 16,6 / 51,4 mm | 16 / 295 / 50 mm |
| arány mag / test / udvar (z' = 100 mm) | 0,93 / 0,04 / 0,03 | 0,81 / 0,03 / 0,17 | 0,16 / 0,00 / 0,84 |
| nyom hatótáva R(T) | 864 mm | 5287 mm | 22483 mm |
| delta-udvar aránya | 0,899 % | 0,79 % | 1,3 % |
| csúcs helye / σ_z / σ_r | −9,95 / 23 / 9,2 mm | −15,2 / 27,8 / 9,1 mm | −21,1 / 28,7 / 9,21 mm |

**Proton** (m = 938,3 MeV; polinomfok és paraméterszám: n = 6, 143)

`ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H`

- Zápor mélységprofilja plató: `L(z') ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ·z'/1 m)` (értékek lent).
- Zápor keresztmetszete: `T(r│z') = f_mag·Gauss(r; σ) + f_test·Cauchy(r; R) + f_udvar·e^(−r/λ)/(2πλ(r + 1 mm))`; σ, R, λ és az arányok a mélységgel változnak (lent z' = 100 mm-en).
- Nyom: hatótáv `R(T) = 757 mm · (T/GeV)^1,78 − 36,5 mm`; csillapítás `e^(−z'/266 mm)`; Bragg-felfutás `(1 − z'/R + 0,03)^(−0,804)`; szélesség `σ(z')² = (18,5 mm)² + (285 mm · (z'/1 m)^2,37 / βp)²`; delta-elektron-udvar `e^(−r/81,5 mm)` (aránya lent).
- Folt a hatótáv végén: Gauss, σ_z = 76,3 mm, σ_r = 81,1 mm.
- Csúcs a belépésnél: Gauss (hely és szélességek lent). Távoli udvar: `H ∝ σ(z'/10 mm)·e^(−z'/962 mm)/(r + 1 cm)`.

| | p = 1 GeV/c | p = 5 GeV/c | p = 20 GeV/c |
|---|---:|---:|---:|
| T (mozgási energia) | 0,433 GeV | 4,15 GeV | 19,1 GeV |
| energiarész: zápor | 14,9 % | 73,6 % | 81,1 % |
| energiarész: nyom | 60,5 % | 21,1 % | 16,8 % |
| energiarész: csúcs | 23,6 % | 4,87 % | 2,12 % |
| energiarész: folt | 0,671 % | < 0,01 % | < 0,01 % |
| energiarész: távoli udvar | 0,383 % | 0,408 % | < 0,01 % |
| zápor kezdete z_s / felfutás w1 | −7,12 / 123 mm | 22,3 / 86,3 mm | 80,7 / 80,2 mm |
| zápor vége ℓ / lecsengés w2 | 132 / 198 mm | 863 / 378 mm | 1330 / 76,2 mm |
| lassú meredekség κ | −3,02 /m | −2,72 /m | −2,65 /m |
| mag σ / Cauchy R / udvar λ (z' = 100 mm) | 84,4 / 121 / 227 mm | 58,8 / 71 / 207 mm | 38,9 / 77,6 / 203 mm |
| arány mag / test / udvar (z' = 100 mm) | 0,08 / 0,87 / 0,05 | 0,08 / 0,80 / 0,13 | 0,13 / 0,76 / 0,10 |
| nyom hatótáva R(T) | 135 mm | 9485 mm | 143749 mm |
| delta-udvar aránya | 3,94 % | 33,4 % | 20,4 % |
| csúcs helye / σ_z / σ_r | −3,17 / 13,7 / 10,1 mm | −7,75 / 22,1 / 11,5 mm | −18,5 / 23,8 / 11,6 mm |

**Pion** (m = 139,6 MeV; polinomfok és paraméterszám: n = 3, 101)

`ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H`

- Zápor mélységprofilja plató: `L(z') ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ·z'/1 m)` (értékek lent).
- Zápor keresztmetszete: `T(r│z') = f_mag·Gauss(r; σ) + f_test·Cauchy(r; R) + f_udvar·e^(−r/λ)/(2πλ(r + 1 mm))`; σ, R, λ és az arányok a mélységgel változnak (lent z' = 100 mm-en).
- Nyom: hatótáv `R(T) = 952 mm · (T/GeV)^1,26 − 26,9 mm`; csillapítás `e^(−z'/339 mm)`; Bragg-felfutás `(1 − z'/R + 0,03)^(−0,526)`; szélesség `σ(z')² = (20,1 mm)² + (135 mm · (z'/1 m)^1,72 / βp)²`; delta-elektron-udvar `e^(−r/69,4 mm)` (aránya lent).
- Folt a hatótáv végén: Gauss, σ_z = 103 mm, σ_r = 130 mm.
- Csúcs a belépésnél: Gauss (hely és szélességek lent). Távoli udvar: `H ∝ σ(z'/10 mm)·e^(−z'/701 mm)/(r + 1 cm)`.

| | p = 1 GeV/c | p = 5 GeV/c | p = 20 GeV/c |
|---|---:|---:|---:|
| T (mozgási energia) | 0,87 GeV | 4,86 GeV | 19,9 GeV |
| energiarész: zápor | 43 % | 68,1 % | 68,5 % |
| energiarész: nyom | 43 % | 26,7 % | 28,9 % |
| energiarész: csúcs | 13,4 % | 4,93 % | 2,64 % |
| energiarész: folt | 0,0701 % | < 0,01 % | < 0,01 % |
| energiarész: távoli udvar | 0,52 % | 0,176 % | 0,0112 % |
| zápor kezdete z_s / felfutás w1 | 12,8 / 111 mm | 44,8 / 80,8 mm | 84,7 / 66,6 mm |
| zápor vége ℓ / lecsengés w2 | 56,9 / 167 mm | 224 / 423 mm | 1207 / 596 mm |
| lassú meredekség κ | −1,99 /m | −2 /m | −2,03 /m |
| mag σ / Cauchy R / udvar λ (z' = 100 mm) | 68 / 96,3 / 216 mm | 71,9 / 67,5 / 206 mm | 59,1 / 67,8 / 196 mm |
| arány mag / test / udvar (z' = 100 mm) | 0,27 / 0,66 / 0,07 | 0,13 / 0,77 / 0,11 | 0,12 / 0,72 / 0,16 |
| nyom hatótáva R(T) | 772 mm | 6980 mm | 41375 mm |
| delta-udvar aránya | 16,3 % | 19,3 % | 14,9 % |
| csúcs helye / σ_z / σ_r | −5,3 / 20,8 / 11,1 mm | −7,04 / 23,9 / 11,6 mm | −9,26 / 23,5 / 11,4 mm |

<!-- KEPLETEK:END -->

## 3. Pontosság

Teszt-NLL, report-mérce (zárójelben a paraméterszám). A „régi képlet × Ā” a report képlete ugyanígy, sablonnal újraillesztve – ez az összehasonlítás alapja.

| részecske | régi képlet (report) | régi képlet × Ā | új formula | Δ teszt | Δ s2 | háló (report / finomhangolt) | rés bezárva |
|---|---:|---:|---:|---:|---:|---:|---:|
| elektron | 14,8479 (33) | 14,8406 (27) | **14,8167** (75) | −0,0239 ± 0,0009 | −0,0236 ± 0,0003 | 14,8068 / 14,8067 | 70 % |
| pion | 18,1563 (27) | 18,1420 (39) | **18,1235** (101) | −0,0186 ± 0,0007 | −0,0212 ± 0,0004 | 18,1221 / 18,1208 | 88 % |
| proton | 18,2536 (45) | 18,2331 (45) | **18,1940** (143) | −0,0391 ± 0,0016 | −0,0409 ± 0,0007 | 18,1922 / 18,1898 | 90 % |
| müon | 16,0899 (59) | 16,0846 (59) | **16,0120** (129) | −0,0726 ± 0,0011 | −0,0739 ± 0,0007 | 16,0046 / 16,0034 | 89 % |

Olvasás: a proton új formulája eseményenként 0,039 nat-tal jobb a régi képletnél, és a régi képlet–háló távolság 90 %-át hozza be,
miközben a hálónak 881–3841 paramétere van. 3D-mércén a rés bezárva: elektron 68 %, proton 88 %, müon 83 % (pionnál a 3D-háló
degenerált, lásd 9.). A 2. fokú, kisebb változat legfeljebb 0,0012 nat-tal rosszabb.

## 4. Alakok (burok-testek)

Egy cellában „V (V_szelet) / c”: V a test térfogata literben az adattartományban (−200 < z' < 1450 mm, r < 700 mm), V_szelet ebből a
detektorba eső rész, c a tesztzáporok hányada, amelyekben az energia ≥ 95 %-a a testen belül van. Mind a 20 impulzus-bin átlaga.
Pl. „1423 (1302) / 0,98”: 1423 L, ebből 1302 L a detektorban, a záporok 98 %-ában az energia legalább 95 %-a bent van.

| alak | elektron | pion | proton | müon |
|---|---:|---:|---:|---:|
| 98 % – csepp (report) | 26,1 (24,6) / 0,98 | 1486 (1362) / 0,98 | 1616 (1469) / 0,98 | – |
| 98 % – csepp, újraillesztve | 25,9 (24,4) / 0,98 | 1478 (1335) / 0,98 | 1600 (1428) / 0,98 | 46,5 (43,5) / 0,98 |
| 98 % – **szuper-tojás** | 25,9 (24,2) / 0,98 | **1423** (1302) / 0,98 | **1544** (1395) / 0,98 | **37,7** (35,8) / 0,98 |
| 98 % – kalibrált izoszint (új formula) | 31,7 (30,7) / 0,98 | 1491 (1368) / 0,98 | 1596 (1449) / 0,98 | 48,9 (44,9) / 0,98 |
| r95 – új formula (E_in = átlagos bent lévő energiahányad) | 5,6 (5,6) / 0,43, E_in 0,93 | 389 (370) / 0,76, E_in 0,95 | 494 (467) / 0,77, E_in 0,95 | 9,3 (8,8) / 0,05, E_in 0,75 |

**Szuper-tojás:** `r(z') = A·(1 − |τ|^n)^(1/m)`, ahol `τ = (z_c − z')/a1` a legszélesebb pont (z_c) előtt, `(z' − z_c)/a2` mögötte;
A a legnagyobb sugár, a1 / a2 az elülső / hátsó félhossz, n, m az alak „szögletessége”. Binenként 6 paraméter, közvetlenül a
záporokra illesztve (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van). Példa (proton, 4–5 GeV):
z_c ≈ 175 mm, a1 ≈ 338 mm, a2 ≈ 1231 mm, n ≈ 2,8, m ≈ 1,7, A ≈ 719 mm. A report cseppje: `r = A·t^(1/2)(1 − t)^(3/2)/g_max`.
A report saját számai pontosan visszajönnek.

## 5. Ellenőrző mérések

**Átlagos profil** (s2; S × 10⁻³, mélység / sugár; mindegyik modell 3D-mércén tanítva):

| részecske | padló | régi képlet | új formula | háló |
|---|---:|---:|---:|---:|
| elektron | 0,16 / 0,11 | 0,76 / 6,51 | 0,88 / 4,97 | 0,56 / 4,55 |
| müon | 0,07 / 0,08 | 5,28 / 15,9 | 4,19 / **1,88** | 0,31 / 2,09 |
| proton | 0,18 / 0,07 | 3,03 / 2,82 | 0,53 / 2,64 | 0,25 / 2,48 |
| pion | 0,17 / 0,05 | 2,25 / 1,06 | 0,77 / 0,98 | (degenerált) |

Sugárirányban a formula a hálóhoz közeli (müonnál jobb), mélységben a háló jobb. A profil tömbjében (p ≥ 2 GeV, 0 < z' < 1300 mm) a
formula energiával súlyozott négyzetes relatív eltérése: proton 2,1 %, pion 1,5 %, elektron 2,9 %, müon 5,8 %. A belépés előtti
(z' < 0) energiát összességében visszaadja (pl. proton 4,1 % az adat 3,9 %-ával szemben), de a szögfüggése túl gyenge (6.). Eltér még:
a müon legmélyebb ~13 cm-e és mélységi meredeksége, az elektron mély farka.

**Ki adja le az energiát** (s2, a szimuláció találatonkénti adatai; 0–20 GeV tartományai):

| részecske | a beeső részecske saját energiája | neutronok leányai | r ≥ 300 mm: adat (modell) | ebből neutronból / azonnali (< 5 ns) |
|---|---:|---:|---:|---:|
| elektron | 1–4 % | ≈ 0,1 % | 0,56–0,91 % (0,56–0,65 %) | 1–2 % / 99 % |
| müon | 79–91 % | 0–1 % | 0,05–0,09 % (0,06–0,07 %) | 2–20 % / 72–96 % |
| proton | 5–50 % | 18–25 % | 3,5–7,9 % (3,7–8,5 %) | 51–75 % / 41–60 % |
| pion | 5–26 % | 11–16 % | 4,1–6,4 % (4,4–6,5 %) | 45–60 % / 55–62 % |

Az elektron távoli udvara tehát azonnali elektromágneses energia (a modell ~30 %-kal alulbecsli); hadronoknál a távoli energiát a zápor
exponenciális udvara viszi. A müon nyoma a saját ionizációja és a közeli delta-elektronok.

**A detektor anyagából várt értékek** (`scripts/lfhcal_expectations.py`; Highland-képlet, CSDA-hatótáv, PDG): a müon szélesedésének
kitevője (1,41) egyezik, mert a 40 mm-es acél előlap 1,5 alá viszi; c 17 %-kal nagyobb (a nyom menti energiaveszteség megmagyarázza);
a proton hatótáva T = 0,5–1 GeV-en 4–16 %-on belül van; a csillapítási hosszak a magkölcsönhatási hosszak ~1,3-szorosai; az elektron
zápormaximumának energiafüggése gyengébb a vártnál.

## 6. Szögfüggő kiegészítés – nem modellpredikció

**Mi ez.** A végleges formula nem tudja, milyen szögben érkezett a részecske. Ebben a kiegészítésben a beesési szöget (θ) eseményenként
**bemenetként megadjuk** – a szimulációból ismert –, és néhány belépési paraméter (a csúcs súlya, helye, szélessége, a zápor és a nyom
kezdete) a szöggel lineárisan változhat: `P(x, u) = P(x) + β·u`, u = 1/cosθ − 1,1; részecskénként 6–8 új paraméter (β).

**Miért nem modellpredikció.** A szögfüggést a formula fizikája nem jósolja meg: a β együtthatókat az adatra illesztettük, ez empirikus
korrekció. A formula maga, és a 2–5. pont minden száma szög nélküli. A hálót sem tanítottuk szöggel, ezért az alábbi összevetés vele
nem azonos feltételű.

| részecske | 3D teszt-NLL: végleges formula → szögfüggő | Δ a végleges formulához (s2) | háló (szög nélkül) |
|---|---:|---:|---:|
| elektron | 14,8072 → 14,8047 | −0,002 | 14,7976 |
| müon | 16,0144 → 15,9690 | −0,045 | 16,0000 |
| proton | 18,1660 → 18,1576 | −0,008 | 18,1614 |
| pion | 18,1014 → 18,0928 | −0,009 | (degenerált) |

A nyereség hadronoknál az első ~2 rétegből jön: a belépési csúcs súlya nagy szögnél 2–6-szorosára nő. A belépés előtti energia
szögtrendjének hibáját felére csökkenti (pl. proton, θ ≈ 0,25 → 0,57 rad: adat 1,8 → 4,7 %, végleges formula 2,8 → 3,8 %, szögfüggő
2,2 → 4,0 %). Müonnál az illesztés átrendezi a tagokat, ott a tagok fizikai értelmezése nem érvényes. Ezért a szögfüggő változat nem része
a végleges formulának.

## 7. Továbbiak

**Időbeli szétválasztás** (proton, pion; s1 teszt). Az adat cellánként csak az első találat idejét tárolja; „késleltetett” az a cella,
amelyet az esemény első találata után 5 ns-nál később talált el először valami. A késleltetett cellák energiája eseményenként átlagosan
8–11 % (proton) ill. 6–8 % (pion) – ez alsó becslés –, és ~2,5–3-szor szélesebb (átlagos sugár ~240–250 mm a gyors ~85–97 mm-ével
szemben), ~115–150 mm-rel mélyebben. A formula egyetlen exponenciális udvara (λ ≈ 200–215 mm) a gyors (≈ 135–160 mm) és a késleltetett
(≈ 410–600 mm) illesztés közé esik. Következő lépés: két udvar-tag (gyors és késleltetett).

**Záporonkénti ingadozás** (s1; főkomponens-elemzés az eseményenkénti profilokon). Az első 3 főkomponens a záporonkénti variancia
elektronnál és müonnál ~3/4-ét, hadronoknál 2 GeV felett csak ~40–50 %-át írja le. Egy főkomponensekből épített egyszerű generátor
10 komponenssel is a padló többszöröse. A formula átlag; eseménygenerátorhoz külön, nem-Gauss ingadozásmodell kell.

## 8. Megállapítások

1. A detektor-elfogadás (× Ā) paraméter nélkül javít; a hálók ennek egy részét „kúpként” tanulták meg.
2. Az új formula mind a négy részecskénél jobb a régi képletnél (−0,019 … −0,073 nat), és a hálóhoz képesti rés 70–90 %-át zárja be.
3. A kiegészítő tagok közül a müon többszörös szórása és az elektron távoli udvara hozta a legtöbbet; a további tagok és a szimbolikus
   regresszióval (PySR) talált korrekciók ≤ 0,0025 nat-ot adnak – a formula itt telítődik.
4. Irodalom: a Bock-féle kétgammás mélységprofil és a GFlash EM-laterális profilja rosszabb; kaloriméter-záporra 2025–26-ból nem
   találtunk új zárt parametrizációt (a terület generatív hálók felé ment: CaloHadronic, AllShowers, CaloDiT-2; a mérce a CaloChallenge
   szeparációs ereje, arXiv:2410.21611).
5. A report cseppje közel optimális burok; a szuper-tojás 3,5–3,8 %-kal kisebb, a kalibrált izoszint nem kisebb (a 98 %-os kritériumot
   a szélsőséges záporok döntik el).

## 9. Korlátok

- Egyik mérce sem normál pontosan a műszerezett cellákra (~0,2 nat); ezért csak azonos módon tanított modellek vannak összevetve.
- A tagok kiválasztásánál a teszthalmazon végzett diagnosztika is szerepet játszott; ezért a tiszta ellenőrzés a független s2 minta.
- A pion 3D-hálója degenerált (tömege gyakorlatilag a detektor-szeleten kívül van), ezért a pion 3D-háló számai nem megbízhatók.
- A belépés előtti energia szögfüggése túl gyenge; a 100 ns után érkező energia kimarad; a formula nem ad záporonkénti ingadozást.
- A cellákat egyenletes 48,6 × 48,6 × 20,5 mm-es dobozként kezeli a modell (a modulhatárokon a valódi osztás nagyobb).
- Hadronoknál a nyom kiegészítőinek paraméterei effektívek.

## 10. Fájlok, futtatás

- Modell: `particle_shape.py`; betöltés: `shapelab.build_final`. Végleges eredmények: `results/particles/<részecske>/test_v3.json`,
  `s2_v3.json`, `shapes.json`; részecskénkénti leírás: `results/particles/<részecske>/formula.md`.
- Futtatás: `scripts/run_all.sh` → `scripts/variants_v2.py`, `variants_v3.py` → `scripts/run_round3.sh` (végleges formula, teszt, s2,
  alakok, ellenőrzés); ellenőrzés: `scripts/verify_results.py`.
- Mérések: `scripts/profile_metrics.py`, `truth_decomposition.py`, `front_region.py`, `lfhcal_expectations.py`; szögfüggő kiegészítés:
  `scripts/theta_entrance.py`; időbeli szétválasztás és ingadozás: `scripts/time_split.py`, `fluctuations_pca.py`.
- Ez a leírás PDF-ben: `README.pdf` (`scripts/build_readme_pdf.sh`). Ábrás protonelemzés: `08_shape3d_proton.ipynb`. Az adat-gyorsítótár (`cache/`) első futáskor a CSV-kből újraépül.
