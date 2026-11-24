# A végleges formula – pion

Tömeg m = 0.13957 GeV; változat `plateau_track+all+all4`; a paraméterek az x = (ln(T/GeV) − v0)/vs változó polinomjai, T = √(p² + m²) − m. Illesztés: a tanítóhalmaz találataira, eseményenkénti likelihooddal (`shapelab.fit_fast`), a fokszám és a tagok a validáción választva.

## Röviden

- **Mi ez:** Mint a proton: nyom az első magkölcsönhatásig, utána hadronzápor. A csillapítási hossz ≈ 1,31 λ_π, a protonénál hosszabb, ahogy a pion kölcsönhatási hossza is. A nyom kiegészítőinek paraméterei effektívek.
- **Pontosság (teszt-NLL, report-mérce, kisebb = jobb):** új formula 18,1235, az ugyanígy illesztett baseline × Ā 18,1420, a finomhangolt háló 18,1208. A baseline és a háló közti rés 88 %-át zárja be (3D-mérce: 63 %; független s2 minta: 90 % / 69 %).
- **A detektorba eső energia megoszlása, 1 GeV → 20 GeV** (3D-mérce): zápor 43 % → 68 %; nyom 43 % → 29 %; csúcs 13 % → 3 %; folt 0,07 % → < 0,01 %; távoli udvar 0,52 % → 0,011 %.
- **Nyom:** hatótáv R ≈ 772 mm 1 GeV-en (kitevő k_R ≈ 1,26); csillapítási hossz λ_att ≈ 339 mm; szélesedés c ≈ 135 mm, k ≈ 1,72; Bragg-kitevő q ≈ 0,53.
- **Keresztmetszet z' = 100 mm-en, 1 → 20 GeV:** Gauss-mag σ ≈ 68 → 59 mm, Cauchy-test R ≈ 96 → 68 mm, udvar λ ≈ 216 mm; arányok (mag / test / udvar) 0,27 / 0,66 / 0,07 → 0,12 / 0,72 / 0,16.
- **98 %-os burkok** (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van; 20 bin átlaga): szuper-tojás 1423 L (1302 L a detektorban, lefedés 0,98); csepp 1478 L (1335 L a detektorban, lefedés 0,98); kalibrált izoszint 1491 L (1368 L a detektorban, lefedés 0,98).
- **Átlagos profilok** (független s2 minta, szeparációs erő × 10⁻³, mélység / sugár, kisebb = jobb): új formula 0,77 / 0,98, háló 1,79 / 0,75, régi képlet 2,25 / 1,06; statisztikai padló 0,17 / 0,05. A pion 3D-hálója degenerált (tömege gyakorlatilag a detektor-szeleten kívül van), az értékei nem megbízhatók.
- **A belépés előtti tartomány (z' < 0) energiája a beesési szöggel** (θ ≈ 0,25 → 0,57 rad, p ≥ 2 GeV): adat 1,4 → 4,1 %, új formula 2,3 → 3,0 %, háló 2,2 → 7,2 %.
- **Szimulációs igazság** (s2, 0–20 GeV négy impulzus-tartományában): a beeső részecske saját lerakódása 5–26 %, a neutronok leányaié 11–16 %; a 100 ns után érkező (a fitből kihagyott) energia 2,7–5,1 %. Az r ≥ 300 mm-es tartomány az energia 4,11–6,41 %-a (a modellben 4,36–6,51 %), ennek 45–60 %-a neutronból, 22–34 %-a elektromágneses és 55–62 %-a azonnali (< 5 ns); a modellben ennek 0–7 %-át viszi a távoli udvar tag.
- **Szögfüggő kiegészítés – nem modellpredikció.** A beesési szöget eseményenként bemenetként megadjuk, és 8 belépési paraméter (csúcs, zápor- és nyomkezdet) a szöggel lineárisan változhat: P + β·(1/cosθ − 1,1). A β-kat az adatra illesztettük (empirikus korrekció, a formula fizikája nem jósolja); a többi szám ezen a kártyán szög nélküli. 3D teszt-NLL: 18,1012 → 18,0928 (s2: -0,0088 ± 0,0002 nat/esemény). A belépés előtti energia (θ ≈ 0,25 → 0,57 rad): adat 1,4 → 4,1 %, végleges formula 2,3 → 3,0 %, szögfüggő 1,7 → 3,6 %.
- **Időben szétválasztva** (cellánként: gyors az a cella, amelyet az első találat után < 5 ns-on belül, késleltetett, amelyet később talált el először valami; a késleltetett hányad így alsó becslés): a késleltetett cellák energiája eseményenként átlagosan 6,3–8,3 %. Az 5–20 GeV-es átlagos sugár / r90 (modell): gyors 85–87 / 183–205 mm, késleltetett 231–252 / 476–508 mm; az udvar hossza z' = 100 mm-en: gyors 155–158 mm, késleltetett 570–598 mm.
- **Záporonkénti ingadozás** (PCA az eseményenkénti profilokon): az első 3 főkomponens a variancia 42–54 %-a. Az eseményenkénti mennyiségek szeparációs ereje (× 10⁻³, a két félminta padlója 3–14): bármely átlag-modell 923 (binelési állandó), 3 főkomponens Gauss-együtthatókkal 154–278, 10 főkomponens a tanító együtthatóiból 39–97.

## A képlet

ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H — minden tag külön a (z', r) rácsra normálva (σ(·) = logisztikus függvény; ami ennél a részecskénél nincs bekapcsolva, az kimarad):

- **Zápor** S = L(z')·T(r | z'). Plató-mélységprofil: L ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ z'/1 m); gamma: L ∝ s^(α−1)·e^(−(α−1) s/t_max), s ≈ z' − z0. Keresztmetszet: T = f1·e^(−r²/2σ²)/(2πσ²) + f2·R²/(π(r² + R²)²) + f3·e^(−r/λ)/(2πλ(r + 1 mm)); ln σ, ln R, ln λ és a keverési logitok lineárisak a mélységben (σ-nál és a logitoknál négyzetes taggal is, ha „görbület” be van kapcsolva); minden mélységben az r < 700 mm korongra normálva.
- **Nyom** U ∝ σ((z' − z_t0)/4 mm)·σ((R(T) − z')/w_R)·e^(−κ_t z'/R)·e^(−z'/λ_att)·(1 − z'/R + 0,03)^(−q) × [(1 − f)·Gauss(r; σ(z')) + f·e^(−r/l)/(2πl(r + 1 mm))], ahol R(T) = R1·T^(k_R) − R0 (hatótáv), w_R = w·R + 3 mm, σ(z')² = σ0² + (c·(z'/1 m)^k / βp)² (többszörös szórás), βp = (T² + 2Tm)/(T + m).
- **Csúcs** a belépésnél: D ∝ e^(−(z' − z_sp)²/2σ_z²)·e^(−r²/2σ_r²); **folt** a hatótáv végén: E ∝ e^(−(z' − R(T))²/2σ_ez²)·e^(−r²/2σ_er²); **távoli udvar**: H ∝ σ(z'/10 mm)·e^(−z'/L)·e^(−r/R)/(r + 10 mm).
- **Súlyok**: w_távoli = σ(a_far), w_folt = σ(a_end)·(1 − w_távoli); a zápor, a nyom és a csúcs a maradékon osztozik (a csúcs a nyomhoz kötve: f1 = d1/(d1 + L_t), L_t a nyom hossza).
- Minden nyers paraméter P(x) = Σ c_k x^k, x = (ln(T/GeV) − v0)/vs, T = √(p² + m²) − m; „log_” = a mennyiség logaritmusa, „logit_” = arány logitja.

Tagok: zápor (plató-mélységprofil × [Gauss-mag + Cauchy-test + e^(−r/λ)/r udvar]) + nyom (hatótáv, csillapítás) + csúcs a belépésnél + távoli udvar + folt a hatótáv végén + többszörös szórás (a nyom szélesedése) + delta-elektron-udvar a nyom körül + Bragg-felfutás + négyzetes mélységtagok a keresztmetszetben.

## Hogyan olvasd a táblázatokat

- Oszlopok: az impulzus p [GeV]; minden sor egy illesztett mennyiség értéke ennél az impulzusnál.
- **w …**: a komponens súlya a vágatlan forgástestben. **a detektorban: …**: a komponens aránya a detektorba (a ferde belépési sík mögé) eső energiában – ez a fizikailag értelmezhető energiaarány.
- **z_s, w1, ℓ, w2, κ** (plató) ill. **z0, t_max, α−1** (gamma): a zápor mélységprofilja. **σ mag, R Cauchy, λ udvar** és **f mag / test / udvar**: a keresztmetszet három tagjának szélessége és aránya az adott mélységben.
- **R(T)**: a nyom hossza (hatótáv); **σ0**: a nyom szélessége a belépésnél; **c_ms, k_ms**: a többszörös szórás miatti szélesedés; **λ_att**: csillapítási hossz; **f, l delta-udvar**: a nyom körüli udvar aránya és hossza; **q Bragg**: a Bragg-felfutás kitevője; **csúcs …, folt …**: a két Gauss-folt helye és szélessége; **távoli udvar L, R**: mélységi és sugárirányú hossz (R → ∞: sugárirányban nincs levágás).
- Két illesztés van: a **3D-mérce** (eseményenként a detektor-szeletre normálva; ez írja le a forgástestet) és a **report-mérce** (× Ā; ebből számolódik az r95- és az izoszint-alakot). A kettő értékei kissé eltérnek.

## 3D-mérce (eseményenként a detektor-szeletre normálva) – ez a forgástest

Fokszám n = 3, 101 paraméter; v0 = -0.3259, vs = 3.3146. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.38 | 0.87 | 1.87 | 4.86 | 9.86 | 19.9 |
| w zápor | 0.339 | 0.506 | 0.615 | 0.688 | 0.702 | 0.682 |
| w nyom | 0.383 | 0.324 | 0.267 | 0.235 | 0.24 | 0.272 |
| w csúcs | 0.27 | 0.165 | 0.115 | 0.0755 | 0.0576 | 0.0462 |
| w folt | 0.00422 | 0.000528 | 8e-05 | 9.1e-06 | 2.34e-06 | 8.22e-07 |
| w távoli udvar | 0.00434 | 0.00435 | 0.00359 | 0.00171 | 0.000589 | 0.000117 |
| a detektorban: zápor | 0.219 | 0.43 | 0.583 | 0.681 | 0.702 | 0.685 |
| a detektorban: nyom | 0.533 | 0.43 | 0.329 | 0.267 | 0.263 | 0.289 |
| a detektorban: csúcs | 0.237 | 0.134 | 0.084 | 0.0493 | 0.0351 | 0.0264 |
| a detektorban: folt | 0.0058 | 0.000701 | 2.28e-05 | 2.13e-06 | 5.27e-07 | 1.8e-07 |
| a detektorban: távoli udvar | 0.00544 | 0.0052 | 0.004 | 0.00176 | 0.000583 | 0.000112 |
| z_s [mm] | 2.12 | 12.8 | 25.1 | 44.8 | 63.1 | 84.7 |
| w1 [mm] | 132 | 111 | 95.7 | 80.8 | 72.6 | 66.6 |
| ℓ [mm] | 37 | 56.9 | 96 | 224 | 488 | 1207 |
| w2 [mm] | 94.6 | 167 | 266 | 423 | 533 | 596 |
| κ [1/m] | -2 | -1.99 | -1.99 | -2 | -2.01 | -2.03 |
| σ mag, z'=100 [mm] | 61.4 | 68 | 71.9 | 71.9 | 67.4 | 59.1 |
| R Cauchy, z'=100 [mm] | 118 | 96.3 | 80.1 | 67.5 | 64.5 | 67.8 |
| λ udvar, z'=100 [mm] | 220 | 216 | 211 | 206 | 201 | 196 |
| f mag / test / udvar, z'=100 | 0.43 / 0.52 / 0.05 | 0.27 / 0.66 / 0.07 | 0.18 / 0.74 / 0.09 | 0.13 / 0.77 / 0.11 | 0.11 / 0.76 / 0.13 | 0.12 / 0.72 / 0.16 |
| σ mag, z'=600 [mm] | 139 | 168 | 188 | 194 | 180 | 152 |
| R Cauchy, z'=600 [mm] | 403 | 259 | 174 | 113 | 89.1 | 77.7 |
| λ udvar, z'=600 [mm] | 596 | 532 | 472 | 400 | 349 | 302 |
| f mag / test / udvar, z'=600 | 0.06 / 0.85 / 0.09 | 0.04 / 0.87 / 0.09 | 0.03 / 0.87 / 0.10 | 0.04 / 0.85 / 0.12 | 0.06 / 0.80 / 0.14 | 0.12 / 0.71 / 0.18 |
| R(T) hatótáv [mm] | 254 | 772 | 2064 | 6980 | 1.71e+04 | 4.14e+04 |
| σ0 nyom [mm] | 20.1 | 20.1 | 20.1 | 20.1 | 20.1 | 20.1 |
| λ_att [mm] | 339 | 339 | 339 | 339 | 339 | 339 |
| c_ms [mm] | 135 | 135 | 135 | 135 | 135 | 135 |
| k_ms | 1.72 | 1.72 | 1.72 | 1.72 | 1.72 | 1.72 |
| f delta-udvar | 0.125 | 0.163 | 0.187 | 0.193 | 0.178 | 0.149 |
| l delta-udvar [mm] | 69.4 | 69.4 | 69.4 | 69.4 | 69.4 | 69.4 |
| q Bragg | 0.526 | 0.526 | 0.526 | 0.526 | 0.526 | 0.526 |
| csúcs z [mm] | -4.85 | -5.3 | -5.94 | -7.04 | -8.07 | -9.26 |
| csúcs σ_z [mm] | 18.4 | 20.8 | 22.6 | 23.9 | 24 | 23.5 |
| csúcs σ_r [mm] | 10.6 | 11.1 | 11.4 | 11.6 | 11.6 | 11.4 |
| folt σ_z [mm] | 103 | 103 | 103 | 103 | 103 | 103 |
| folt σ_r [mm] | 130 | 130 | 130 | 130 | 130 | 130 |
| távoli udvar L [mm] | 701 | 701 | 701 | 701 | 701 | 701 |
| távoli udvar R [mm] | 8.7e+05 | 8.7e+05 | 8.7e+05 | 8.7e+05 | 8.7e+05 | 8.7e+05 |

Együtthatók: `formula_coefficients_slab.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).

## report-mérce (× Ā; ebből számolódik az r95- és az izoszint-alakot)

Fokszám n = 3, 101 paraméter; v0 = -0.3259, vs = 3.3146. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.38 | 0.87 | 1.87 | 4.86 | 9.86 | 19.9 |
| w zápor | 0.325 | 0.484 | 0.594 | 0.683 | 0.722 | 0.748 |
| w nyom | 0.485 | 0.411 | 0.333 | 0.267 | 0.238 | 0.22 |
| w csúcs | 0.178 | 0.097 | 0.0632 | 0.0388 | 0.0277 | 0.0204 |
| w folt | 0.00558 | 0.000763 | 0.000133 | 1.9e-05 | 5.87e-06 | 2.44e-06 |
| w távoli udvar | 0.00527 | 0.0073 | 0.00925 | 0.0113 | 0.012 | 0.0118 |
| a detektorban: zápor | 0.184 | 0.374 | 0.53 | 0.654 | 0.707 | 0.739 |
| a detektorban: nyom | 0.591 | 0.501 | 0.388 | 0.293 | 0.253 | 0.229 |
| a detektorban: csúcs | 0.213 | 0.116 | 0.0723 | 0.0418 | 0.0289 | 0.0209 |
| a detektorban: folt | 0.00671 | 0.000934 | 3.55e-05 | 4.25e-06 | 1.27e-06 | 5.16e-07 |
| a detektorban: távoli udvar | 0.00577 | 0.00801 | 0.00972 | 0.0111 | 0.0115 | 0.011 |
| z_s [mm] | 6.15 | 17.4 | 30.3 | 50.7 | 69.2 | 90.8 |
| w1 [mm] | 140 | 117 | 98.4 | 80.2 | 70 | 62.5 |
| ℓ [mm] | 25.9 | 36.2 | 59.3 | 154 | 421 | 1529 |
| w2 [mm] | 103 | 175 | 285 | 524 | 815 | 1257 |
| κ [1/m] | -1.97 | -2.04 | -2.11 | -2.19 | -2.22 | -2.22 |
| σ mag, z'=100 [mm] | 63.8 | 60.3 | 57 | 50.5 | 43.2 | 34.1 |
| R Cauchy, z'=100 [mm] | 130 | 107 | 91.3 | 79.2 | 76.3 | 79.3 |
| λ udvar, z'=100 [mm] | 90.7 | 89.3 | 87.5 | 84.7 | 82.3 | 79.5 |
| f mag / test / udvar, z'=100 | 0.39 / 0.57 / 0.04 | 0.22 / 0.73 / 0.05 | 0.12 / 0.81 / 0.07 | 0.07 / 0.84 / 0.09 | 0.06 / 0.81 / 0.13 | 0.06 / 0.76 / 0.19 |
| σ mag, z'=600 [mm] | 108 | 139 | 191 | 303 | 431 | 603 |
| R Cauchy, z'=600 [mm] | 398 | 256 | 173 | 111 | 85.6 | 70.9 |
| λ udvar, z'=600 [mm] | 329 | 299 | 266 | 218 | 183 | 150 |
| f mag / test / udvar, z'=600 | 0.12 / 0.83 / 0.05 | 0.06 / 0.88 / 0.06 | 0.03 / 0.88 / 0.09 | 0.02 / 0.82 / 0.16 | 0.02 / 0.70 / 0.28 | 0.02 / 0.51 / 0.47 |
| R(T) hatótáv [mm] | 245 | 755 | 2046 | 7035 | 1.74e+04 | 4.27e+04 |
| σ0 nyom [mm] | 20.1 | 20.1 | 20.1 | 20.1 | 20.1 | 20.1 |
| λ_att [mm] | 297 | 297 | 297 | 297 | 297 | 297 |
| c_ms [mm] | 117 | 117 | 117 | 117 | 117 | 117 |
| k_ms | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 |
| f delta-udvar | 0.144 | 0.288 | 0.363 | 0.233 | 0.0626 | 0.00488 |
| l delta-udvar [mm] | 78.4 | 78.4 | 78.4 | 78.4 | 78.4 | 78.4 |
| q Bragg | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 | 0.52 |
| csúcs z [mm] | 5.3 | 5.43 | 5.57 | 5.75 | 5.89 | 6.04 |
| csúcs σ_z [mm] | 4.77 | 5.05 | 4.99 | 4.48 | 3.88 | 3.19 |
| csúcs σ_r [mm] | 10.6 | 11.2 | 11.5 | 11.6 | 11.5 | 11.1 |
| folt σ_z [mm] | 99 | 99 | 99 | 99 | 99 | 99 |
| folt σ_r [mm] | 125 | 125 | 125 | 125 | 125 | 125 |
| távoli udvar L [mm] | 1236 | 1236 | 1236 | 1236 | 1236 | 1236 |
| távoli udvar R [mm] | 2.55e+05 | 2.55e+05 | 2.55e+05 | 2.55e+05 | 2.55e+05 | 2.55e+05 |

Együtthatók: `formula_coefficients_std.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).
