# A végleges formula – müon

Tömeg m = 0.105658 GeV; változat `full_att+all+all4`; a paraméterek az x = (ln(T/GeV) − v0)/vs változó polinomjai, T = √(p² + m²) − m. Illesztés: a tanítóhalmaz találataira, eseményenkénti likelihooddal (`shapelab.fit_fast`), a fokszám és a tagok a validáción választva.

## Röviden

- **Mi ez:** Minimálisan ionizáló nyom: az energia nagy része a nyomban van (a primer saját ionizációja és a közeli delta-elektronok). A nyom a mélységgel a többszörös szórás szerint szélesedik; a Highland-képlet az LFHCal rétegeire a 40 mm-es acél előlappal ≈ 62–63 mm-t és 1,35–1,41-es kitevőt ad. A végén Bragg-felfutás (Bragg–Kleeman: q ≈ 0,43); a hatótáv közel lineáris T-ben.
- **Pontosság (teszt-NLL, report-mérce, kisebb = jobb):** új formula 16,0120, az ugyanígy illesztett baseline × Ā 16,0846, a finomhangolt háló 16,0034. A baseline és a háló közti rés 89 %-át zárja be (3D-mérce: 83 %; független s2 minta: 90 % / 84 %).
- **A detektorba eső energia megoszlása, 1 GeV → 20 GeV** (3D-mérce): zápor 2 % → 1 %; nyom 91 % → 93 %; csúcs 6 % → 6 %; folt 0,24 % → < 0,01 %; távoli udvar 0,085 % → 0,094 %.
- **Nyom:** hatótáv R ≈ 864 mm 1 GeV-en (kitevő k_R ≈ 1,02); csillapítási hossz λ_att ≈ 1211 mm; szélesedés c ≈ 74 mm, k ≈ 1,41; Bragg-kitevő q ≈ 0,42.
- **Keresztmetszet z' = 100 mm-en, 1 → 20 GeV:** Gauss-mag σ ≈ 11 → 16 mm, Cauchy-test R ≈ 10 → 295 mm, udvar λ ≈ 52 mm; arányok (mag / test / udvar) 0,93 / 0,04 / 0,03 → 0,16 / 0,00 / 0,84.
- **98 %-os burkok** (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van; 20 bin átlaga): szuper-tojás 37,7 L (35,8 L a detektorban, lefedés 0,98); csepp 46,5 L (43,5 L a detektorban, lefedés 0,98); kapszula 38,7 L (36,8 L a detektorban, lefedés 0,98); kalibrált izoszint 48,9 L (44,9 L a detektorban, lefedés 0,98).
- **Átlagos profilok** (független s2 minta, szeparációs erő × 10⁻³, mélység / sugár, kisebb = jobb): új formula 4,19 / 1,88, háló 0,31 / 2,09, régi képlet 5,28 / 15,92; statisztikai padló 0,07 / 0,08.
- **A belépés előtti tartomány (z' < 0) energiája a beesési szöggel** (θ ≈ 0,25 → 0,57 rad, p ≥ 2 GeV): adat 1,0 → 4,2 %, új formula 2,1 → 2,3 %, háló 2,1 → 2,4 %.
- **Szimulációs igazság** (s2, 0–20 GeV négy impulzus-tartományában): a beeső részecske saját lerakódása 79–91 %, a neutronok leányaié 0–1 %; a 100 ns után érkező (a fitből kihagyott) energia 0,0–1,7 %. Az r ≥ 300 mm-es tartomány az energia 0,05–0,09 %-a (a modellben 0,06–0,07 %), ennek 2–20 %-a neutronból, 11–36 %-a elektromágneses és 72–96 %-a azonnali (< 5 ns); a modellben ennek 89–91 %-át viszi a távoli udvar tag.
- **Szögfüggő kiegészítés – nem modellpredikció.** A beesési szöget eseményenként bemenetként megadjuk, és 7 belépési paraméter (csúcs, zápor- és nyomkezdet) a szöggel lineárisan változhat: P + β·(1/cosθ − 1,1). A β-kat az adatra illesztettük (empirikus korrekció, a formula fizikája nem jósolja); a többi szám ezen a kártyán szög nélküli. 3D teszt-NLL: 16,0148 → 15,9690 (s2: -0,0457 ± 0,0005 nat/esemény). A belépés előtti energia (θ ≈ 0,25 → 0,57 rad): adat 1,0 → 4,2 %, végleges formula 2,1 → 2,3 %, szögfüggő 1,0 → 2,5 %. Müonnál az illesztés átrendezi a tagokat, ott a tagok fizikailag nem értelmezhetők.
- **Záporonkénti ingadozás** (PCA az eseményenkénti profilokon): az első 3 főkomponens a variancia 73–78 %-a. Az eseményenkénti mennyiségek szeparációs ereje (× 10⁻³, a két félminta padlója 3–15): bármely átlag-modell 923 (binelési állandó), 3 főkomponens Gauss-együtthatókkal 137–196, 10 főkomponens a tanító együtthatóiból 52–76.

## A képlet

ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H — minden tag külön a (z', r) rácsra normálva (σ(·) = logisztikus függvény; ami ennél a részecskénél nincs bekapcsolva, az kimarad):

- **Zápor** S = L(z')·T(r | z'). Plató-mélységprofil: L ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ z'/1 m); gamma: L ∝ s^(α−1)·e^(−(α−1) s/t_max), s ≈ z' − z0. Keresztmetszet: T = f1·e^(−r²/2σ²)/(2πσ²) + f2·R²/(π(r² + R²)²) + f3·e^(−r/λ)/(2πλ(r + 1 mm)); ln σ, ln R, ln λ és a keverési logitok lineárisak a mélységben (σ-nál és a logitoknál négyzetes taggal is, ha „görbület” be van kapcsolva); minden mélységben az r < 700 mm korongra normálva.
- **Nyom** U ∝ σ((z' − z_t0)/4 mm)·σ((R(T) − z')/w_R)·e^(−κ_t z'/R)·e^(−z'/λ_att)·(1 − z'/R + 0,03)^(−q) × [(1 − f)·Gauss(r; σ(z')) + f·e^(−r/l)/(2πl(r + 1 mm))], ahol R(T) = R1·T^(k_R) − R0 (hatótáv), w_R = w·R + 3 mm, σ(z')² = σ0² + (c·(z'/1 m)^k / βp)² (többszörös szórás), βp = (T² + 2Tm)/(T + m).
- **Csúcs** a belépésnél: D ∝ e^(−(z' − z_sp)²/2σ_z²)·e^(−r²/2σ_r²); **folt** a hatótáv végén: E ∝ e^(−(z' − R(T))²/2σ_ez²)·e^(−r²/2σ_er²); **távoli udvar**: H ∝ σ(z'/10 mm)·e^(−z'/L)·e^(−r/R)/(r + 10 mm).
- **Súlyok**: w_távoli = σ(a_far), w_folt = σ(a_end)·(1 − w_távoli); a zápor, a nyom és a csúcs a maradékon osztozik (a csúcs a nyomhoz kötve: f1 = d1/(d1 + L_t), L_t a nyom hossza).
- Minden nyers paraméter P(x) = Σ c_k x^k, x = (ln(T/GeV) − v0)/vs, T = √(p² + m²) − m; „log_” = a mennyiség logaritmusa, „logit_” = arány logitja.

Tagok: zápor (gamma-mélységprofil × [Gauss-mag + Cauchy-test + e^(−r/λ)/r udvar]) + nyom (hatótáv, csillapítás) + csúcs a belépésnél + távoli udvar + folt a hatótáv végén + többszörös szórás (a nyom szélesedése) + delta-elektron-udvar a nyom körül + Bragg-felfutás + négyzetes mélységtagok a keresztmetszetben.

## Hogyan olvasd a táblázatokat

- Oszlopok: az impulzus p [GeV]; minden sor egy illesztett mennyiség értéke ennél az impulzusnál.
- **w …**: a komponens súlya a vágatlan forgástestben. **a detektorban: …**: a komponens aránya a detektorba (a ferde belépési sík mögé) eső energiában – ez a fizikailag értelmezhető energiaarány.
- **z_s, w1, ℓ, w2, κ** (plató) ill. **z0, t_max, α−1** (gamma): a zápor mélységprofilja. **σ mag, R Cauchy, λ udvar** és **f mag / test / udvar**: a keresztmetszet három tagjának szélessége és aránya az adott mélységben.
- **R(T)**: a nyom hossza (hatótáv); **σ0**: a nyom szélessége a belépésnél; **c_ms, k_ms**: a többszörös szórás miatti szélesedés; **λ_att**: csillapítási hossz; **f, l delta-udvar**: a nyom körüli udvar aránya és hossza; **q Bragg**: a Bragg-felfutás kitevője; **csúcs …, folt …**: a két Gauss-folt helye és szélessége; **távoli udvar L, R**: mélységi és sugárirányú hossz (R → ∞: sugárirányban nincs levágás).
- Két illesztés van: a **3D-mérce** (eseményenként a detektor-szeletre normálva; ez írja le a forgástestet) és a **report-mérce** (× Ā; ebből számolódik az r95- és az izoszint-alakot). A kettő értékei kissé eltérnek.

## 3D-mérce (eseményenként a detektor-szeletre normálva) – ez a forgástest

Fokszám n = 6, 129 paraméter; v0 = -0.6642, vs = 3.6546. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.405 | 0.9 | 1.9 | 4.9 | 9.89 | 19.9 |
| w zápor | 0.0449 | 0.0681 | 0.08 | 0.0659 | 0.0383 | 0.0142 |
| w nyom | 0.791 | 0.819 | 0.814 | 0.818 | 0.833 | 0.844 |
| w csúcs | 0.156 | 0.11 | 0.104 | 0.115 | 0.128 | 0.141 |
| w folt | 0.00761 | 0.00217 | 0.000515 | 4.27e-05 | 3.14e-06 | 8.6e-08 |
| w távoli udvar | 0.000983 | 0.000859 | 0.000843 | 0.000928 | 0.000999 | 0.000922 |
| a detektorban: zápor | 0.000801 | 0.0242 | 0.0698 | 0.0611 | 0.0359 | 0.0139 |
| a detektorban: nyom | 0.891 | 0.909 | 0.872 | 0.88 | 0.903 | 0.925 |
| a detektorban: csúcs | 0.0984 | 0.0638 | 0.0574 | 0.0579 | 0.0596 | 0.0597 |
| a detektorban: folt | 0.00857 | 0.00239 | 0.000114 | 8.84e-06 | 6.56e-07 | 1.81e-08 |
| a detektorban: távoli udvar | 0.00099 | 0.000852 | 0.000844 | 0.00093 | 0.00101 | 0.000942 |
| z0 [mm] | -44.8 | -48.6 | -52 | -57.1 | -62.5 | -71 |
| t_max [mm] | 2.23 | 5.36 | 17 | 90.9 | 234 | 240 |
| α−1 | 0.248 | 0.0747 | 0.0243 | 0.00916 | 0.0107 | 0.0508 |
| σ mag, z'=100 [mm] | 10.5 | 11.2 | 11.5 | 11.7 | 12.6 | 16 |
| R Cauchy, z'=100 [mm] | 9.75 | 9.51 | 10.3 | 16.6 | 42.7 | 295 |
| λ udvar, z'=100 [mm] | 52.4 | 52.3 | 52 | 51.4 | 50.7 | 50 |
| f mag / test / udvar, z'=100 | 0.94 / 0.03 / 0.02 | 0.93 / 0.04 / 0.03 | 0.91 / 0.04 / 0.05 | 0.81 / 0.03 / 0.17 | 0.57 / 0.01 / 0.43 | 0.16 / 0.00 / 0.84 |
| σ mag, z'=600 [mm] | 13.3 | 14 | 14.3 | 14.4 | 15.4 | 19.5 |
| R Cauchy, z'=600 [mm] | 9.33 | 7.18 | 5.39 | 4.41 | 5.92 | 18.9 |
| λ udvar, z'=600 [mm] | 61.5 | 60.5 | 58.4 | 54.3 | 50.5 | 46.2 |
| f mag / test / udvar, z'=600 | 0.97 / 0.01 / 0.02 | 0.96 / 0.01 / 0.03 | 0.94 / 0.00 / 0.06 | 0.83 / 0.00 / 0.17 | 0.54 / 0.00 / 0.46 | 0.12 / 0.00 / 0.88 |
| R(T) hatótáv [mm] | 335 | 864 | 1951 | 5287 | 1.1e+04 | 2.25e+04 |
| σ0 nyom [mm] | 15 | 15 | 15 | 15 | 15 | 15 |
| λ_att [mm] | 1211 | 1211 | 1211 | 1211 | 1211 | 1211 |
| c_ms [mm] | 74.4 | 74.4 | 74.4 | 74.4 | 74.4 | 74.4 |
| k_ms | 1.41 | 1.41 | 1.41 | 1.41 | 1.41 | 1.41 |
| f delta-udvar | 0.0109 | 0.00899 | 0.00806 | 0.0079 | 0.00902 | 0.013 |
| l delta-udvar [mm] | 51.4 | 51.4 | 51.4 | 51.4 | 51.4 | 51.4 |
| q Bragg | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 | 0.42 |
| csúcs z [mm] | -8.21 | -9.95 | -12 | -15.2 | -18 | -21.1 |
| csúcs σ_z [mm] | 19.8 | 23 | 25.5 | 27.8 | 28.7 | 28.7 |
| csúcs σ_r [mm] | 9.34 | 9.2 | 9.13 | 9.1 | 9.13 | 9.21 |
| folt σ_z [mm] | 73.1 | 73.1 | 73.1 | 73.1 | 73.1 | 73.1 |
| folt σ_r [mm] | 85.6 | 85.6 | 85.6 | 85.6 | 85.6 | 85.6 |
| távoli udvar L [mm] | 1309 | 1309 | 1309 | 1309 | 1309 | 1309 |
| távoli udvar R [mm] | 2.93e+08 | 2.93e+08 | 2.93e+08 | 2.93e+08 | 2.93e+08 | 2.93e+08 |

Együtthatók: `formula_coefficients_slab.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).

## report-mérce (× Ā; ebből számolódik az r95- és az izoszint-alakot)

Fokszám n = 6, 129 paraméter; v0 = -0.6642, vs = 3.6546. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.405 | 0.9 | 1.9 | 4.9 | 9.89 | 19.9 |
| w zápor | 0.029 | 0.0555 | 0.0836 | 0.0979 | 0.0757 | 0.0328 |
| w nyom | 0.872 | 0.884 | 0.867 | 0.856 | 0.878 | 0.92 |
| w csúcs | 0.0876 | 0.0555 | 0.0473 | 0.0445 | 0.0454 | 0.0465 |
| w folt | 0.0101 | 0.00338 | 0.00112 | 0.000182 | 2.66e-05 | 1.65e-06 |
| w távoli udvar | 0.00138 | 0.00108 | 0.000972 | 0.001 | 0.00107 | 0.00103 |
| a detektorban: zápor | 0.000132 | 0.0132 | 0.0734 | 0.0867 | 0.0675 | 0.0301 |
| a detektorban: nyom | 0.899 | 0.925 | 0.876 | 0.866 | 0.884 | 0.921 |
| a detektorban: csúcs | 0.0891 | 0.0574 | 0.0494 | 0.0464 | 0.0472 | 0.0481 |
| a detektorban: folt | 0.0104 | 0.00352 | 0.000224 | 3.47e-05 | 5.05e-06 | 3.12e-07 |
| a detektorban: távoli udvar | 0.00127 | 0.00101 | 0.000919 | 0.000943 | 0.00101 | 0.000964 |
| z0 [mm] | -39.8 | -43.1 | -46.7 | -53.1 | -61 | -73.9 |
| t_max [mm] | 0.765 | 3.55 | 45 | 1.27e+04 | 1.73e+07 | 2.17e+12 |
| α−1 | 0.249 | 0.0693 | 0.0292 | 0.0198 | 0.0311 | 0.108 |
| σ mag, z'=100 [mm] | 11 | 11.4 | 11.5 | 11.6 | 12.4 | 15.1 |
| R Cauchy, z'=100 [mm] | 23.9 | 9.49 | 5.49 | 5.77 | 16.3 | 209 |
| λ udvar, z'=100 [mm] | 50.8 | 50.6 | 50.3 | 49.7 | 49.2 | 48.5 |
| f mag / test / udvar, z'=100 | 0.87 / 0.11 / 0.02 | 0.88 / 0.09 / 0.03 | 0.88 / 0.06 / 0.06 | 0.84 / 0.02 / 0.14 | 0.75 / 0.00 / 0.25 | 0.47 / 0.00 / 0.53 |
| σ mag, z'=600 [mm] | 14.7 | 15.1 | 15.1 | 15.1 | 15.9 | 19.1 |
| R Cauchy, z'=600 [mm] | 4.59 | 1.42 | 0.537 | 0.251 | 0.318 | 1.56 |
| λ udvar, z'=600 [mm] | 59.9 | 58.4 | 56.2 | 52.4 | 49 | 45.2 |
| f mag / test / udvar, z'=600 | 0.94 / 0.00 / 0.06 | 0.93 / 0.00 / 0.06 | 0.92 / 0.00 / 0.08 | 0.86 / 0.00 / 0.14 | 0.71 / 0.00 / 0.29 | 0.35 / 0.00 / 0.65 |
| R(T) hatótáv [mm] | 329 | 852 | 1930 | 5253 | 1.09e+04 | 2.25e+04 |
| σ0 nyom [mm] | 14.9 | 14.9 | 14.9 | 14.9 | 14.9 | 14.9 |
| λ_att [mm] | 1353 | 1353 | 1353 | 1353 | 1353 | 1353 |
| c_ms [mm] | 75.2 | 75.2 | 75.2 | 75.2 | 75.2 | 75.2 |
| k_ms | 1.42 | 1.42 | 1.42 | 1.42 | 1.42 | 1.42 |
| f delta-udvar | 0.0084 | 0.00762 | 0.00721 | 0.00701 | 0.00721 | 0.0083 |
| l delta-udvar [mm] | 52.9 | 52.9 | 52.9 | 52.9 | 52.9 | 52.9 |
| q Bragg | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 |
| csúcs z [mm] | 5.94 | 6.05 | 6.1 | 6.11 | 6.07 | 6 |
| csúcs σ_z [mm] | 4.19 | 5.13 | 5.65 | 5.61 | 5.08 | 4.26 |
| csúcs σ_r [mm] | 9.66 | 9.54 | 9.47 | 9.44 | 9.46 | 9.51 |
| folt σ_z [mm] | 63.4 | 63.4 | 63.4 | 63.4 | 63.4 | 63.4 |
| folt σ_r [mm] | 70.8 | 70.8 | 70.8 | 70.8 | 70.8 | 70.8 |
| távoli udvar L [mm] | 1456 | 1456 | 1456 | 1456 | 1456 | 1456 |
| távoli udvar R [mm] | 2.8e+08 | 2.8e+08 | 2.8e+08 | 2.8e+08 | 2.8e+08 | 2.8e+08 |

Együtthatók: `formula_coefficients_std.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).
