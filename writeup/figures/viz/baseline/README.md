# UI-Baseline

Aufruf: `node writeup/figures/viz/capture.mjs --url http://localhost:5173`

1600 × 1000 CSS-Pixel, DPR 2 (3200 × 2000 PNG); Chromium 156.0.8078.4, ANGLE Metal. Jeder Zustand wird frisch mit URL-Pose geladen und mindestens 1.5 s beruhigt. Keine Anwendungscode-Änderungen. Trend: 100-ms-Pose-Schritte, Aufnahme während laufender Bewegung nach ca. 4 s; Timing und Hundertstel können variieren. Kritischer Abstand ist durch depth ≤ 22 auf ca. 0.36 mm begrenzt. Backenöffnung und laterale Auslenkung verändern die tatsächliche Distanz.

- 01_far_overview.png: Fern; Scope in Bereitschaft; Pose {"depth":14.36,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 7.99 mm; Ansicht Overview.
- 02_near_overview.png: Nah; Scope aktiv; Pose {"depth":18.36,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 4.00 mm; Ansicht Overview.
- 03_warning_overview.png: Warnbereich; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- 04_critical_overview.png: Kritischer Abstand (Tiefenlimit 22 mm); Pose {"depth":22,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 0.36 mm; Ansicht Overview.
- 05_warning_surgeon.png: Warnbereich aus Operateursperspektive; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Surgeon.
- 06_warning_sagittal.png: Warnbereich im Sagittalschnitt; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Sagittal.
- 07_warning_fundus.png: Warnbereich in Fundusansicht; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Fundus.
- 08_approach_trend.png: Annäherung über 4 s von ca. 5 auf 1.5 mm; Pose {"depth":"20.86 → 21.1225","u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.58 → 1.23 (Messungen vor/nach Screenshot; siehe PNG) mm; im PNG visuell bestätigt: 1.58 mm, 0.88 mm/s, Kontakt in ≈ 1.8 s; Ansicht Overview.
- 09_nodata_live.png: Live tracking ohne Messdaten; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand — mm; Ansicht Overview.
- 10_display_menu_open.png: Display-Menü geöffnet; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- 11_legend_and_nav.png: Legende und Navigation; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- 12_jaws_closed.png: Backen geschlossen; Pose {"depth":18.36,"u":0,"v":0,"half":0.02,"roll":0}; angezeigter Abstand 3.99 mm; Ansicht Overview.
- 13_jaws_wide.png: Backen maximal geöffnet; Pose {"depth":18.36,"u":0,"v":0,"half":0.4,"roll":0}; angezeigter Abstand 4.09 mm; Ansicht Overview.
- 14_lateral_sway.png: Seitliche Auslenkung; Schatten/Landepunkt versetzt; Pose {"depth":17.42,"u":0.5,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 2.00 mm; Ansicht Overview.
- crop_approach.png: Warnbereich, Panel-Ausschnitt; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- crop_distance.png: Warnbereich, Panel-Ausschnitt; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- crop_instrument.png: Warnbereich, Panel-Ausschnitt; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.
- crop_scope.png: Warnbereich, Panel-Ausschnitt; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.20 mm; Ansicht Overview.

JavaScript-Laufzeitfehler: keine.

Visuell geprüft: 01, 03, 08, 14 und crop_scope; Auge, Instrument, Schatten und Scope sichtbar. Anfangs SwiftShader-Screenshot-Timeout; mit ANGLE Metal behoben. Trendprognose kann bei langsamen Frames verpasst werden; bis zu vier Wiederholungen im Harness.
