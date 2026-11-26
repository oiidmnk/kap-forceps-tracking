# Quantitative Farbanalyse der Statusfarben

## Methode

sRGB-Tokens aus `services/viz/src/workstation.css:4–34`, abgeglichen mit `theme.js:4–6`; Wörter/Formen aus `ui.js:10–27`. Panel #131b24, Inset #10171f, Hintergrund #0c1117. Machado et al. (2009), Schweregrad 1.0 und 0.6: Matrizen auf lineares sRGB, anschließend Gamut-Clipping auf [0,1] und sRGB-Kodierung. CIELAB D65/2°, CIEDE2000 (kL=kC=kH=1); OKLab euklidisch auf nativer 0–1-Skala (kein ×100). Keine chromatische Adaptation. ΔE00 <10: schwer, 10–<20: schwach, ≥20: darüber; heuristische Vergleichsgrenzen, keine klinische Validierung. Kontrast nach WCAG relativer Leuchtdichte; bei CVD sind **auch die Flächen simuliert**. CVD-Kontrastwerte sind modellierte Werte, keine zusätzliche WCAG-Konformitätsprüfung. Opake Chips: Hintergrund beeinflusst den Kontrast, nicht den intrinsischen Farbabstand.

Referenz: Machado, Oliveira & Fernandes, IEEE TVCG 15(6), 1291–1298 (2009), doi:10.1109/TVCG.2009.113; Matrizen https://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html. DE00-Code gegen drei Sharma-Referenzpaare geprüft.

## Farbabstände

| Simulation | Paar | ΔE00 | OKLab ΔE | Einordnung |
|---|---|---:|---:|---|
| Normal | safe–warn | 33.60 | 0.1615 | ≥20 |
| Normal | safe–danger | 60.25 | 0.2514 | ≥20 |
| Normal | warn–danger | 35.84 | 0.1675 | ≥20 |
| Protan 1.0 | safe–warn | 13.18 | 0.0765 | schwach |
| Protan 1.0 | safe–danger | 14.04 | 0.1391 | schwach |
| Protan 1.0 | warn–danger | 21.90 | 0.1613 | ≥20 |
| Deutan 1.0 | safe–warn | 18.30 | 0.1141 | schwach |
| Deutan 1.0 | safe–danger | 7.77 | 0.0411 | schwer |
| Deutan 1.0 | warn–danger | 13.17 | 0.1076 | schwach |
| Tritan 1.0 | safe–warn | 48.90 | 0.2143 | ≥20 |
| Tritan 1.0 | safe–danger | 62.39 | 0.2963 | ≥20 |
| Tritan 1.0 | warn–danger | 14.07 | 0.1302 | schwach |
| Protan 0.6 | safe–warn | 17.36 | 0.0833 | schwach |
| Protan 0.6 | safe–danger | 27.96 | 0.1212 | ≥20 |
| Protan 0.6 | warn–danger | 25.39 | 0.1498 | ≥20 |
| Deutan 0.6 | safe–warn | 19.90 | 0.1080 | schwach |
| Deutan 0.6 | safe–danger | 21.59 | 0.0691 | ≥20 |
| Deutan 0.6 | warn–danger | 17.94 | 0.1177 | schwach |
| Tritan 0.6 | safe–warn | 39.51 | 0.1800 | ≥20 |
| Tritan 0.6 | safe–danger | 60.70 | 0.2658 | ≥20 |
| Tritan 0.6 | warn–danger | 24.26 | 0.1418 | ≥20 |

## Leuchtdichtekontrast (Verhältnisse :1)

| Simulation | safe/Panel | warn/Panel | danger/Panel | safe–warn | safe–danger | warn–danger |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 8.34 | 9.36 | 6.51 | 1.12 | 1.28 | 1.44 |
| Protan 1.0 | 8.83 | 8.55 | 5.33 | 1.03 | 1.66 | 1.60 |
| Deutan 1.0 | 7.99 | 9.88 | 7.29 | 1.24 | 1.10 | 1.35 |
| Tritan 1.0 | 8.26 | 9.23 | 6.27 | 1.12 | 1.32 | 1.47 |
| Protan 0.6 | 8.63 | 8.86 | 5.79 | 1.03 | 1.49 | 1.53 |
| Deutan 0.6 | 8.12 | 9.67 | 6.98 | 1.19 | 1.16 | 1.38 |
| Tritan 0.6 | 8.40 | 9.44 | 6.50 | 1.12 | 1.29 | 1.45 |

## 3D-Kontext und Grenzen

Retina #8b3a4a, Opazität 0.13 (`scene/EyeGlobe.jsx:130–137`), Instrumentenschatten #14070a, Opazität 0.6 (`scene/InstrumentShadow.jsx:16–17,64`), Scope-Wand #654451 (`scene/ProximityWall.jsx:117`). Diese Materialwerte sind keine fertigen Pixel: Licht, Tone Mapping, Überlagerung und Hintergrund bestimmen die tatsächlichen 3D-Kontraste. Als **rein illustrative sRGB-Alpha-Mischung** über #0c1117 ergibt Retina #1d161e. Keine Übertragung der Panel-WCAG-Werte auf 3D.

## Neutrale Schlussfolgerung

Die opaken Statusfarben liefern im Normalmodell 6.51–9.36:1 gegen das Panel und überschreiten damit 4.5:1 für Text.
Bei vollständiger Rot-Grün-Fehlsichtigkeit rücken mehrere Statuspaare unter die heuristischen Grenzen; die exakten Werte stehen oben.
Die Kontraste zwischen Statusfarben sind gering; Leuchtdichte allein bietet keine robuste dreistufige Statuskodierung.
Der Hauptstatus besitzt Wörter und verschiedene Formen (●/▲/■); diese Redundanz erhält die semantische Unterscheidung auch bei ähnlichen Farben.
L/R-Balken vermitteln Abstand zusätzlich durch Position; dies unterstützt die Interpretation, ersetzt aber keine explizite Statusbezeichnung.
Die einzelnen 3D-Tipkugeln und der Statusverlauf der Szenenlegende kodieren die Klassen nur farblich und sind daher besonders betroffen.
Der Szenenalarm ergänzt Farbe durch Bannertext/Form, 2-/3-px-Rahmen und bei Critical einen Puls; reduced-motion schaltet den Puls ab.
Die Simulation ist eine modellbasierte Prüfung von Tokens, kein Nachweis der Wahrnehmbarkeit im OP oder der tatsächlich gerenderten 3D-Pixel.

Abbildung: `status_cvd.png`; 3180×2040 px, 300 dpi, weiße Außenfläche, Original-Panel pro Chip. Alle Zeilen zeigen absichtlich denselben Beispielwert; die Statuswörter sind keine daraus abgeleitete Klassifikation.

## Option: farbfehlsichtigkeitsrobustere Dreierpalette

Nur als Option: safe #ffffff, warn #56b4e9, danger #e69f00. In diesem Modell liegen alle Paarabstände bei vollständiger CVD über 25; dafür werden die gewohnten Grün/Gelb/Rot-Assoziationen aufgegeben. Wörter und Formen bleiben erforderlich. Keine Entscheidung oder Änderung an der UI.

| Simulation | Paar | ΔE00 | OKLab ΔE |
|---|---|---:|---:|
| Normal | safe–warn | 28.60 | 0.2903 |
| Normal | safe–danger | 34.08 | 0.2933 |
| Normal | warn–danger | 53.61 | 0.2713 |
| Protan 1.0 | safe–warn | 25.45 | 0.2463 |
| Protan 1.0 | safe–danger | 35.11 | 0.3240 |
| Protan 1.0 | warn–danger | 50.54 | 0.2391 |
| Deutan 1.0 | safe–warn | 29.82 | 0.2956 |
| Deutan 1.0 | safe–danger | 32.75 | 0.2826 |
| Deutan 1.0 | warn–danger | 54.26 | 0.2625 |
| Tritan 1.0 | safe–warn | 30.23 | 0.2907 |
| Tritan 1.0 | safe–danger | 31.28 | 0.2771 |
| Tritan 1.0 | warn–danger | 53.64 | 0.2622 |
| Protan 0.6 | safe–warn | 26.28 | 0.2600 |
| Protan 0.6 | safe–danger | 34.53 | 0.3107 |
| Protan 0.6 | warn–danger | 49.22 | 0.2410 |
| Deutan 0.6 | safe–warn | 28.92 | 0.2898 |
| Deutan 0.6 | safe–danger | 33.18 | 0.2868 |
| Deutan 0.6 | warn–danger | 51.63 | 0.2577 |
| Tritan 0.6 | safe–warn | 30.09 | 0.3025 |
| Tritan 0.6 | safe–danger | 31.05 | 0.2741 |
| Tritan 0.6 | warn–danger | 46.43 | 0.2602 |

Normaler Panelkontrast safe / warn / danger: 17.36:1 / 7.52:1 / 7.71:1.

Visuelle Prüfung der PNG durchgeführt: alle zwölf Chips, Wörter und Formen lesbar, keine abgeschnittenen Beschriftungen. Matplotlib-Rekonstruktion der UI-Farben und Informationshierarchie; kein Browser-Screenshot und keine pixelgenaue Typografiereplik.
