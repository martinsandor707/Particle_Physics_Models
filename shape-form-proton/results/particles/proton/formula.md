# A végleges formula – proton

Tömeg m = 0.938272 GeV; változat `plateau_track+all+all4`; a paraméterek az x = (ln(T/GeV) − v0)/vs változó polinomjai, T = √(p² + m²) − m. Illesztés: a tanítóhalmaz találataira, eseményenkénti likelihooddal (`shapelab.fit_fast`), a fokszám és a tagok a validáción választva.

## Röviden

- **Mi ez:** Kis impulzuson megálló proton (nyom + csúcs + folt), nagyobb impulzuson hadronzápor. A hatótáv T = 0,5–1 GeV-en 4–16 %-on belül egyezik a CSDA-hatótávval (az előlap után); a k_R kitevő globális, a CSDA helyi kitevője itt 1,3–1,6. A csillapítási hossz ≈ 1,26 λ_I. A nyom szélesedése, delta-udvara és Bragg-kitevője itt effektív: a hadronmag egy részét is ezek írják le.
- **Pontosság (teszt-NLL, report-mérce, kisebb = jobb):** új formula 18,1940, az ugyanígy illesztett baseline × Ā 18,2331, a finomhangolt háló 18,1898. A baseline és a háló közti rés 90 %-át zárja be (3D-mérce: 88 %; független s2 minta: 91 % / 89 %).
- **A detektorba eső energia megoszlása, 1 GeV → 20 GeV** (3D-mérce): zápor 15 % → 81 %; nyom 61 % → 17 %; csúcs 24 % → 2 %; folt 0,67 % → < 0,01 %; távoli udvar 0,38 % → < 0,01 %.
- **Nyom:** hatótáv R ≈ 135 mm 1 GeV-en (kitevő k_R ≈ 1,78); csillapítási hossz λ_att ≈ 266 mm; szélesedés c ≈ 285 mm, k ≈ 2,37; Bragg-kitevő q ≈ 0,80.
- **Keresztmetszet z' = 100 mm-en, 1 → 20 GeV:** Gauss-mag σ ≈ 84 → 39 mm, Cauchy-test R ≈ 121 → 78 mm, udvar λ ≈ 227 mm; arányok (mag / test / udvar) 0,08 / 0,87 / 0,05 → 0,13 / 0,76 / 0,10.
- **98 %-os burkok** (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van; 20 bin átlaga): szuper-tojás 1544 L (1395 L a detektorban, lefedés 0,98); csepp 1600 L (1428 L a detektorban, lefedés 0,98); kalibrált izoszint 1596 L (1449 L a detektorban, lefedés 0,98).
- **Átlagos profilok** (független s2 minta, szeparációs erő × 10⁻³, mélység / sugár, kisebb = jobb): új formula 0,53 / 2,64, háló 0,25 / 2,48, régi képlet 3,03 / 2,82; statisztikai padló 0,18 / 0,07.
- **A belépés előtti tartomány (z' < 0) energiája a beesési szöggel** (θ ≈ 0,25 → 0,57 rad, p ≥ 2 GeV): adat 1,8 → 4,7 %, új formula 2,8 → 3,8 %, háló 2,7 → 3,6 %.
- **Szimulációs igazság** (s2, 0–20 GeV négy impulzus-tartományában): a beeső részecske saját lerakódása 5–50 %, a neutronok leányaié 18–25 %; a 100 ns után érkező (a fitből kihagyott) energia 3,3–4,3 %. Az r ≥ 300 mm-es tartomány az energia 3,47–7,86 %-a (a modellben 3,69–8,50 %), ennek 51–75 %-a neutronból, 18–29 %-a elektromágneses és 41–60 %-a azonnali (< 5 ns); a modellben ennek 0–8 %-át viszi a távoli udvar tag.
- **Szögfüggő kiegészítés – nem modellpredikció.** A beesési szöget eseményenként bemenetként megadjuk, és 8 belépési paraméter (csúcs, zápor- és nyomkezdet) a szöggel lineárisan változhat: P + β·(1/cosθ − 1,1). A β-kat az adatra illesztettük (empirikus korrekció, a formula fizikája nem jósolja); a többi szám ezen a kártyán szög nélküli. 3D teszt-NLL: 18,1657 → 18,1576 (s2: -0,0073 ± 0,0002 nat/esemény). A belépés előtti energia (θ ≈ 0,25 → 0,57 rad): adat 1,8 → 4,7 %, végleges formula 2,8 → 3,8 %, szögfüggő 2,2 → 4,0 %.
- **Időben szétválasztva** (cellánként: gyors az a cella, amelyet az első találat után < 5 ns-on belül, késleltetett, amelyet később talált el először valami; a késleltetett hányad így alsó becslés): a késleltetett cellák energiája eseményenként átlagosan 8,4–10,9 %. Az 5–20 GeV-es átlagos sugár / r90 (modell): gyors 96–97 / 205–227 mm, késleltetett 238–253 / 476–508 mm; az udvar hossza z' = 100 mm-en: gyors 135–145 mm, késleltetett 410–419 mm.
- **Záporonkénti ingadozás** (PCA az eseményenkénti profilokon): az első 3 főkomponens a variancia 40–66 %-a. Az eseményenkénti mennyiségek szeparációs ereje (× 10⁻³, a két félminta padlója 3–15): bármely átlag-modell 923 (binelési állandó), 3 főkomponens Gauss-együtthatókkal 143–282, 10 főkomponens a tanító együtthatóiból 31–127.

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

Fokszám n = 6, 143 paraméter; v0 = -0.1055, vs = 3.0544. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.125 | 0.433 | 1.27 | 4.15 | 9.11 | 19.1 |
| w zápor | 0.015 | 0.209 | 0.536 | 0.744 | 0.788 | 0.799 |
| w nyom | 0.000791 | 0.504 | 0.326 | 0.179 | 0.154 | 0.151 |
| w csúcs | 0.523 | 0.278 | 0.13 | 0.0729 | 0.0576 | 0.0492 |
| w folt | 0.46 | 0.00584 | 7.88e-05 | 2.67e-06 | 6.07e-07 | 2.24e-07 |
| w távoli udvar | 0.000635 | 0.00355 | 0.00787 | 0.00383 | 0.000411 | 4.75e-06 |
| a detektorban: zápor | 0.0116 | 0.149 | 0.489 | 0.736 | 0.79 | 0.811 |
| a detektorban: nyom | 0.000979 | 0.605 | 0.398 | 0.211 | 0.177 | 0.168 |
| a detektorban: csúcs | 0.568 | 0.236 | 0.104 | 0.0487 | 0.032 | 0.0212 |
| a detektorban: folt | 0.419 | 0.00671 | 9.46e-05 | 6.01e-07 | 1.33e-07 | 4.71e-08 |
| a detektorban: távoli udvar | 0.000925 | 0.00383 | 0.00866 | 0.00408 | 0.000426 | 4.75e-06 |
| z_s [mm] | -11.8 | -7.12 | 2.2 | 22.3 | 45.7 | 80.7 |
| w1 [mm] | 214 | 123 | 92.4 | 86.3 | 88 | 80.2 |
| ℓ [mm] | 51 | 132 | 331 | 863 | 1285 | 1330 |
| w2 [mm] | 205 | 198 | 298 | 378 | 249 | 76.2 |
| κ [1/m] | -3.16 | -3.02 | -2.86 | -2.72 | -2.67 | -2.65 |
| σ mag, z'=100 [mm] | 190 | 84.4 | 63.7 | 58.8 | 52.8 | 38.9 |
| R Cauchy, z'=100 [mm] | 275 | 121 | 82.2 | 71 | 71.9 | 77.6 |
| λ udvar, z'=100 [mm] | 246 | 227 | 215 | 207 | 204 | 203 |
| f mag / test / udvar, z'=100 | 0.08 / 0.89 / 0.04 | 0.08 / 0.87 / 0.05 | 0.08 / 0.84 / 0.08 | 0.08 / 0.80 / 0.13 | 0.09 / 0.78 / 0.14 | 0.13 / 0.76 / 0.10 |
| σ mag, z'=600 [mm] | 651 | 360 | 320 | 342 | 333 | 262 |
| R Cauchy, z'=600 [mm] | 1340 | 424 | 213 | 131 | 104 | 89.3 |
| λ udvar, z'=600 [mm] | 651 | 400 | 292 | 232 | 213 | 206 |
| f mag / test / udvar, z'=600 | 0.19 / 0.79 / 0.02 | 0.18 / 0.80 / 0.03 | 0.13 / 0.82 / 0.05 | 0.09 / 0.81 / 0.11 | 0.07 / 0.78 / 0.15 | 0.07 / 0.76 / 0.17 |
| R(T) hatótáv [mm] | 1.14 | 135 | 1125 | 9485 | 3.85e+04 | 1.44e+05 |
| σ0 nyom [mm] | 18.5 | 18.5 | 18.5 | 18.5 | 18.5 | 18.5 |
| λ_att [mm] | 266 | 266 | 266 | 266 | 266 | 266 |
| c_ms [mm] | 285 | 285 | 285 | 285 | 285 | 285 |
| k_ms | 2.37 | 2.37 | 2.37 | 2.37 | 2.37 | 2.37 |
| f delta-udvar | 0.000969 | 0.0394 | 0.226 | 0.334 | 0.248 | 0.204 |
| l delta-udvar [mm] | 81.5 | 81.5 | 81.5 | 81.5 | 81.5 | 81.5 |
| q Bragg | 0.804 | 0.804 | 0.804 | 0.804 | 0.804 | 0.804 |
| csúcs z [mm] | -6.44 | -3.17 | -3.67 | -7.75 | -12.5 | -18.5 |
| csúcs σ_z [mm] | 8.8 | 13.7 | 18.2 | 22.1 | 23.5 | 23.8 |
| csúcs σ_r [mm] | 8.93 | 10.1 | 10.9 | 11.5 | 11.6 | 11.6 |
| folt σ_z [mm] | 76.3 | 76.3 | 76.3 | 76.3 | 76.3 | 76.3 |
| folt σ_r [mm] | 81.1 | 81.1 | 81.1 | 81.1 | 81.1 | 81.1 |
| távoli udvar L [mm] | 962 | 962 | 962 | 962 | 962 | 962 |
| távoli udvar R [mm] | 5.55e+07 | 5.55e+07 | 5.55e+07 | 5.55e+07 | 5.55e+07 | 5.55e+07 |

Együtthatók: `formula_coefficients_slab.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).

## report-mérce (× Ā; ebből számolódik az r95- és az izoszint-alakot)

Fokszám n = 6, 143 paraméter; v0 = -0.1055, vs = 3.0544. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.125 | 0.433 | 1.27 | 4.15 | 9.11 | 19.1 |
| w zápor | 0.0175 | 0.228 | 0.545 | 0.75 | 0.795 | 0.799 |
| w nyom | 0.000989 | 0.541 | 0.362 | 0.206 | 0.177 | 0.181 |
| w csúcs | 0.432 | 0.221 | 0.084 | 0.0403 | 0.0278 | 0.0205 |
| w folt | 0.55 | 0.00644 | 4.91e-05 | 5.13e-07 | 3.91e-08 | 4.45e-09 |
| w távoli udvar | 0.000232 | 0.00368 | 0.00897 | 0.00397 | 0.000421 | 6.28e-06 |
| a detektorban: zápor | 0.0127 | 0.149 | 0.473 | 0.717 | 0.773 | 0.786 |
| a detektorban: nyom | 0.00106 | 0.6 | 0.421 | 0.234 | 0.196 | 0.193 |
| a detektorban: csúcs | 0.569 | 0.241 | 0.0962 | 0.0449 | 0.03 | 0.0212 |
| a detektorban: folt | 0.417 | 0.00689 | 5.61e-05 | 1.11e-07 | 8.23e-09 | 8.99e-10 |
| a detektorban: távoli udvar | 0.00028 | 0.00366 | 0.00938 | 0.00407 | 0.000421 | 6.03e-06 |
| z_s [mm] | -9.53 | 2.29 | 16.7 | 40.8 | 65.2 | 98.3 |
| w1 [mm] | 215 | 124 | 95.2 | 90.1 | 91 | 80.5 |
| ℓ [mm] | 18.6 | 48.4 | 132 | 542 | 2331 | 2.71e+04 |
| w2 [mm] | 253 | 192 | 330 | 729 | 1804 | 1.43e+04 |
| κ [1/m] | -2.6 | -2.64 | -2.69 | -2.75 | -2.73 | -2.59 |
| σ mag, z'=100 [mm] | 24.8 | 34.8 | 47.1 | 56.5 | 52.4 | 42 |
| R Cauchy, z'=100 [mm] | 286 | 128 | 86.3 | 72.7 | 72.5 | 77.8 |
| λ udvar, z'=100 [mm] | 264 | 244 | 232 | 222 | 218 | 216 |
| f mag / test / udvar, z'=100 | 0.04 / 0.95 / 0.01 | 0.07 / 0.90 / 0.03 | 0.09 / 0.85 / 0.07 | 0.09 / 0.79 / 0.12 | 0.09 / 0.78 / 0.13 | 0.12 / 0.78 / 0.10 |
| σ mag, z'=600 [mm] | 119 | 200 | 298 | 370 | 339 | 261 |
| R Cauchy, z'=600 [mm] | 1587 | 466 | 221 | 129 | 102 | 88.3 |
| λ udvar, z'=600 [mm] | 665 | 421 | 309 | 240 | 214 | 200 |
| f mag / test / udvar, z'=600 | 0.05 / 0.94 / 0.00 | 0.09 / 0.90 / 0.01 | 0.11 / 0.86 / 0.03 | 0.09 / 0.82 / 0.10 | 0.07 / 0.78 / 0.15 | 0.07 / 0.76 / 0.16 |
| R(T) hatótáv [mm] | 1.11 | 139 | 1127 | 9227 | 3.67e+04 | 1.34e+05 |
| σ0 nyom [mm] | 18.9 | 18.9 | 18.9 | 18.9 | 18.9 | 18.9 |
| λ_att [mm] | 249 | 249 | 249 | 249 | 249 | 249 |
| c_ms [mm] | 260 | 260 | 260 | 260 | 260 | 260 |
| k_ms | 2.3 | 2.3 | 2.3 | 2.3 | 2.3 | 2.3 |
| f delta-udvar | 0.00164 | 0.0497 | 0.245 | 0.362 | 0.303 | 0.292 |
| l delta-udvar [mm] | 77.3 | 77.3 | 77.3 | 77.3 | 77.3 | 77.3 |
| q Bragg | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| csúcs z [mm] | 5.8 | 6.2 | 6.22 | 5.89 | 5.48 | 4.94 |
| csúcs σ_z [mm] | 0.753 | 2.39 | 4.5 | 6.12 | 5.98 | 4.96 |
| csúcs σ_r [mm] | 9.52 | 10.4 | 11 | 11.7 | 12 | 12.3 |
| folt σ_z [mm] | 74.3 | 74.3 | 74.3 | 74.3 | 74.3 | 74.3 |
| folt σ_r [mm] | 79.5 | 79.5 | 79.5 | 79.5 | 79.5 | 79.5 |
| távoli udvar L [mm] | 1326 | 1326 | 1326 | 1326 | 1326 | 1326 |
| távoli udvar R [mm] | 2.02e+08 | 2.02e+08 | 2.02e+08 | 2.02e+08 | 2.02e+08 | 2.02e+08 |

Együtthatók: `formula_coefficients_std.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).
