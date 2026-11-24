# A végleges formula – elektron

Tömeg m = 0.000511 GeV; változat `plateau+all+curv`; a paraméterek az x = (ln(T/GeV) − v0)/vs változó polinomjai, T = √(p² + m²) − m. Illesztés: a tanítóhalmaz találataira, eseményenkénti likelihooddal (`shapelab.fit_fast`), a fokszám és a tagok a validáción választva.

## Röviden

- **Mi ez:** Elektromágneses zápor, nyom nincs. Kis impulzuson a belépés körüli csúcs jelentős, nagyobb impulzuson a zápor dominál. A ~1 %-os távoli udvar hozta a legnagyobb javulást a kiegészítő tagok közül; a szimulációs igazság szerint ez azonnali elektromágneses (e±/γ) energia, nem neutron, és a modell 5–20 GeV-en ~30 %-kal alulbecsli.
- **Pontosság (teszt-NLL, report-mérce, kisebb = jobb):** új formula 14,8167, az ugyanígy illesztett baseline × Ā 14,8406, a finomhangolt háló 14,8067. A baseline és a háló közti rés 70 %-át zárja be (3D-mérce: 68 %; független s2 minta: 73 % / 71 %).
- **A detektorba eső energia megoszlása, 1 GeV → 20 GeV** (3D-mérce): zápor 82 % → 99 %; csúcs 17 % → 0,35 %; távoli udvar 0,67 % → 0,95 %.
- **Keresztmetszet z' = 100 mm-en, 1 → 20 GeV:** Gauss-mag σ ≈ 19 → 16 mm, Cauchy-test R ≈ 35 → 21 mm, udvar λ ≈ 40 mm; arányok (mag / test / udvar) 0,74 / 0,13 / 0,13 → 0,85 / 0,02 / 0,13.
- **98 %-os burkok** (a legkisebb test, amelyben a záporok 98 %-ában az energia ≥ 95 %-a bent van; 20 bin átlaga): szuper-tojás 25,9 L (24,2 L a detektorban, lefedés 0,98); csepp 25,9 L (24,4 L a detektorban, lefedés 0,98); kalibrált izoszint 31,7 L (30,7 L a detektorban, lefedés 0,98).
- **Átlagos profilok** (független s2 minta, szeparációs erő × 10⁻³, mélység / sugár, kisebb = jobb): új formula 0,88 / 4,97, háló 0,56 / 4,55, régi képlet 0,76 / 6,51; statisztikai padló 0,16 / 0,11.
- **A belépés előtti tartomány (z' < 0) energiája a beesési szöggel** (θ ≈ 0,25 → 0,57 rad, p ≥ 2 GeV): adat 3,0 → 6,2 %, új formula 3,9 → 4,7 %, háló 4,1 → 5,3 %.
- **Szimulációs igazság** (s2, 0–20 GeV négy impulzus-tartományában): a beeső részecske saját lerakódása 1–4 %, a neutronok leányaié 0 %; a 100 ns után érkező (a fitből kihagyott) energia 0,0–0,1 %. Az r ≥ 300 mm-es tartomány az energia 0,56–0,91 %-a (a modellben 0,56–0,65 %), ennek 1–2 %-a neutronból, 96–98 %-a elektromágneses és 99 %-a azonnali (< 5 ns); a modellben ennek 67–86 %-át viszi a távoli udvar tag.
- **Szögfüggő kiegészítés – nem modellpredikció.** A beesési szöget eseményenként bemenetként megadjuk, és 6 belépési paraméter (csúcs, zápor- és nyomkezdet) a szöggel lineárisan változhat: P + β·(1/cosθ − 1,1). A β-kat az adatra illesztettük (empirikus korrekció, a formula fizikája nem jósolja); a többi szám ezen a kártyán szög nélküli. 3D teszt-NLL: 14,8072 → 14,8047 (s2: -0,0024 ± 0,0001 nat/esemény). A belépés előtti energia (θ ≈ 0,25 → 0,57 rad): adat 3,0 → 6,2 %, végleges formula 3,9 → 4,7 %, szögfüggő 3,4 → 5,7 %.
- **Záporonkénti ingadozás** (PCA az eseményenkénti profilokon): az első 3 főkomponens a variancia 72–74 %-a. Az eseményenkénti mennyiségek szeparációs ereje (× 10⁻³, a két félminta padlója 3–13): bármely átlag-modell 923 (binelési állandó), 3 főkomponens Gauss-együtthatókkal 106–176, 10 főkomponens a tanító együtthatóiból 16–56.

## A képlet

ρ(z', r | p) = w_zápor·S + w_nyom·U + w_csúcs·D + w_folt·E + w_távoli·H — minden tag külön a (z', r) rácsra normálva (σ(·) = logisztikus függvény; ami ennél a részecskénél nincs bekapcsolva, az kimarad):

- **Zápor** S = L(z')·T(r | z'). Plató-mélységprofil: L ∝ σ((z' − z_s)/w1)·σ((ℓ − z')/w2)·e^(κ z'/1 m); gamma: L ∝ s^(α−1)·e^(−(α−1) s/t_max), s ≈ z' − z0. Keresztmetszet: T = f1·e^(−r²/2σ²)/(2πσ²) + f2·R²/(π(r² + R²)²) + f3·e^(−r/λ)/(2πλ(r + 1 mm)); ln σ, ln R, ln λ és a keverési logitok lineárisak a mélységben (σ-nál és a logitoknál négyzetes taggal is, ha „görbület” be van kapcsolva); minden mélységben az r < 700 mm korongra normálva.
- **Nyom** U ∝ σ((z' − z_t0)/4 mm)·σ((R(T) − z')/w_R)·e^(−κ_t z'/R)·e^(−z'/λ_att)·(1 − z'/R + 0,03)^(−q) × [(1 − f)·Gauss(r; σ(z')) + f·e^(−r/l)/(2πl(r + 1 mm))], ahol R(T) = R1·T^(k_R) − R0 (hatótáv), w_R = w·R + 3 mm, σ(z')² = σ0² + (c·(z'/1 m)^k / βp)² (többszörös szórás), βp = (T² + 2Tm)/(T + m).
- **Csúcs** a belépésnél: D ∝ e^(−(z' − z_sp)²/2σ_z²)·e^(−r²/2σ_r²); **folt** a hatótáv végén: E ∝ e^(−(z' − R(T))²/2σ_ez²)·e^(−r²/2σ_er²); **távoli udvar**: H ∝ σ(z'/10 mm)·e^(−z'/L)·e^(−r/R)/(r + 10 mm).
- **Súlyok**: w_távoli = σ(a_far), w_folt = σ(a_end)·(1 − w_távoli); a zápor, a nyom és a csúcs a maradékon osztozik (a csúcs a nyomhoz kötve: f1 = d1/(d1 + L_t), L_t a nyom hossza).
- Minden nyers paraméter P(x) = Σ c_k x^k, x = (ln(T/GeV) − v0)/vs, T = √(p² + m²) − m; „log_” = a mennyiség logaritmusa, „logit_” = arány logitja.

Tagok: zápor (plató-mélységprofil × [Gauss-mag + Cauchy-test + e^(−r/λ)/r udvar]) + csúcs a belépésnél + távoli udvar + négyzetes mélységtagok a keresztmetszetben.

## Hogyan olvasd a táblázatokat

- Oszlopok: az impulzus p [GeV]; minden sor egy illesztett mennyiség értéke ennél az impulzusnál.
- **w …**: a komponens súlya a vágatlan forgástestben. **a detektorban: …**: a komponens aránya a detektorba (a ferde belépési sík mögé) eső energiában – ez a fizikailag értelmezhető energiaarány.
- **z_s, w1, ℓ, w2, κ** (plató) ill. **z0, t_max, α−1** (gamma): a zápor mélységprofilja. **σ mag, R Cauchy, λ udvar** és **f mag / test / udvar**: a keresztmetszet három tagjának szélessége és aránya az adott mélységben.
- **R(T)**: a nyom hossza (hatótáv); **σ0**: a nyom szélessége a belépésnél; **c_ms, k_ms**: a többszörös szórás miatti szélesedés; **λ_att**: csillapítási hossz; **f, l delta-udvar**: a nyom körüli udvar aránya és hossza; **q Bragg**: a Bragg-felfutás kitevője; **csúcs …, folt …**: a két Gauss-folt helye és szélessége; **távoli udvar L, R**: mélységi és sugárirányú hossz (R → ∞: sugárirányban nincs levágás).
- Két illesztés van: a **3D-mérce** (eseményenként a detektor-szeletre normálva; ez írja le a forgástestet) és a **report-mérce** (× Ā; ebből számolódik az r95- és az izoszint-alakot). A kettő értékei kissé eltérnek.

## 3D-mérce (eseményenként a detektor-szeletre normálva) – ez a forgástest

Fokszám n = 3, 64 paraméter; v0 = -0.7756, vs = 3.7713. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.499 | 0.999 | 2 | 5 | 10 | 20 |
| w zápor | 0.664 | 0.742 | 0.82 | 0.915 | 0.962 | 0.983 |
| w csúcs | 0.332 | 0.252 | 0.173 | 0.0762 | 0.0284 | 0.00701 |
| w távoli udvar | 0.00473 | 0.00599 | 0.00731 | 0.0089 | 0.00972 | 0.01 |
| a detektorban: zápor | 0.754 | 0.824 | 0.884 | 0.947 | 0.975 | 0.987 |
| a detektorban: csúcs | 0.24 | 0.17 | 0.108 | 0.0438 | 0.0153 | 0.00351 |
| a detektorban: távoli udvar | 0.00545 | 0.00666 | 0.00781 | 0.00899 | 0.00948 | 0.00951 |
| z_s [mm] | 1.96 | 5.51 | 9.4 | 15.1 | 19.9 | 25.2 |
| w1 [mm] | 18.1 | 19.9 | 21.7 | 24 | 25.4 | 26.3 |
| ℓ [mm] | 33.3 | 42.8 | 56.1 | 82.3 | 112 | 154 |
| w2 [mm] | 57.2 | 60.2 | 62.9 | 65.5 | 66.4 | 66.2 |
| κ [1/m] | -1.68 | -1.44 | -1.26 | -1.16 | -1.19 | -1.33 |
| σ mag, z'=100 [mm] | 20.8 | 19 | 17.6 | 16.5 | 16 | 15.8 |
| R Cauchy, z'=100 [mm] | 39.5 | 34.8 | 31 | 26.7 | 23.7 | 20.7 |
| λ udvar, z'=100 [mm] | 39.5 | 39.5 | 39.5 | 39.5 | 39.5 | 39.5 |
| f mag / test / udvar, z'=100 | 0.69 / 0.19 / 0.12 | 0.74 / 0.13 / 0.13 | 0.78 / 0.08 / 0.14 | 0.82 / 0.04 / 0.13 | 0.84 / 0.03 / 0.13 | 0.85 / 0.02 / 0.13 |
| σ mag, z'=600 [mm] | 2.11 | 1.92 | 1.79 | 1.67 | 1.62 | 1.6 |
| R Cauchy, z'=600 [mm] | 110 | 97.3 | 86.7 | 74.7 | 66.3 | 57.9 |
| λ udvar, z'=600 [mm] | 22.7 | 22.7 | 22.7 | 22.7 | 22.7 | 22.7 |
| f mag / test / udvar, z'=600 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 |
| csúcs z [mm] | -8.25 | -8.84 | -9.54 | -10.6 | -11.6 | -12.6 |
| csúcs σ_z [mm] | 17.9 | 21.9 | 24.5 | 24.6 | 22.2 | 18.3 |
| csúcs σ_r [mm] | 8.9 | 8.88 | 8.94 | 9.15 | 9.4 | 9.74 |
| távoli udvar L [mm] | 767 | 767 | 767 | 767 | 767 | 767 |
| távoli udvar R [mm] | 9.8e+08 | 9.8e+08 | 9.8e+08 | 9.8e+08 | 9.8e+08 | 9.8e+08 |

Együtthatók: `formula_coefficients_slab.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).

## report-mérce (× Ā; ebből számolódik az r95- és az izoszint-alakot)

Fokszám n = 4, 75 paraméter; v0 = -0.7756, vs = 3.7713. A „w” sorok a vágatlan forgástest részarányai, az „a detektorban” sorok a detektorba (a ferde belépési sík mögé) eső energiáé (a szögekre átlagolt elfogadással, Ā-val súlyozva).

| paraméter | p = 0.5 | p = 1 | p = 2 | p = 5 | p = 10 | p = 20 |
|---|---:|---:|---:|---:|---:|---:|
| T [GeV] | 0.499 | 0.999 | 2 | 5 | 10 | 20 |
| w zápor | 0.741 | 0.807 | 0.854 | 0.9 | 0.928 | 0.951 |
| w csúcs | 0.254 | 0.186 | 0.137 | 0.0887 | 0.0603 | 0.0382 |
| w távoli udvar | 0.00498 | 0.00678 | 0.0088 | 0.0111 | 0.0117 | 0.0106 |
| a detektorban: zápor | 0.738 | 0.807 | 0.855 | 0.902 | 0.929 | 0.952 |
| a detektorban: csúcs | 0.258 | 0.187 | 0.137 | 0.0882 | 0.06 | 0.0381 |
| a detektorban: távoli udvar | 0.00463 | 0.00624 | 0.00805 | 0.0101 | 0.0106 | 0.00963 |
| z_s [mm] | 7.35 | 10.2 | 13 | 16.6 | 19.3 | 22 |
| w1 [mm] | 11.2 | 10.4 | 10.5 | 11.5 | 13 | 15.2 |
| ℓ [mm] | 33.1 | 45.4 | 62.7 | 94.5 | 125 | 156 |
| w2 [mm] | 57.7 | 59.6 | 61.1 | 62.1 | 61.6 | 59.5 |
| κ [1/m] | -1.66 | -1.28 | -0.902 | -0.437 | -0.113 | 0.181 |
| σ mag, z'=100 [mm] | 20.6 | 18.6 | 17.4 | 16.4 | 16 | 15.6 |
| R Cauchy, z'=100 [mm] | 39 | 34.1 | 30.3 | 26 | 22.8 | 19.6 |
| λ udvar, z'=100 [mm] | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 | 39.8 |
| f mag / test / udvar, z'=100 | 0.71 / 0.18 / 0.11 | 0.76 / 0.12 / 0.12 | 0.80 / 0.07 / 0.13 | 0.83 / 0.04 / 0.13 | 0.84 / 0.03 / 0.13 | 0.85 / 0.02 / 0.12 |
| σ mag, z'=600 [mm] | 4.54 | 4.11 | 3.83 | 3.62 | 3.52 | 3.43 |
| R Cauchy, z'=600 [mm] | 104 | 90.7 | 80.5 | 69 | 60.7 | 52.2 |
| λ udvar, z'=600 [mm] | 29.3 | 29.3 | 29.3 | 29.3 | 29.3 | 29.3 |
| f mag / test / udvar, z'=600 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | 0.00 / 0.99 / 0.01 | 0.00 / 0.99 / 0.01 | 0.00 / 0.99 / 0.01 |
| csúcs z [mm] | 5.23 | 5.36 | 5.5 | 5.71 | 5.88 | 6.06 |
| csúcs σ_z [mm] | 4.03 | 4.35 | 4.53 | 4.51 | 4.32 | 3.98 |
| csúcs σ_r [mm] | 10.2 | 10.3 | 10.3 | 10.2 | 10.1 | 10 |
| távoli udvar L [mm] | 906 | 906 | 906 | 906 | 906 | 906 |
| távoli udvar R [mm] | 8.94e+09 | 8.94e+09 | 8.94e+09 | 8.94e+09 | 8.94e+09 | 8.94e+09 |

Együtthatók: `formula_coefficients_std.csv` (nyers paraméter = Σ c_k x^k; log_* = a mennyiség logaritmusa, logit_* = arány logitja).
