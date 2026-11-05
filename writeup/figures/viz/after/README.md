# UI screenshots (after)

Current UI; the previous version with identical poses is in ../baseline/.

Aufruf: `node writeup/figures/viz/capture.mjs --url http://localhost:5173 --out /Users/linus/uni/kap-forceps-tracking/writeup/figures/viz/after`

1600 × 1000 CSS-Pixel, DPR 2 (3200 × 2000 PNG); Chromium 156.0.8078.4, ANGLE Metal. Jeder Zustand wird frisch mit URL-Pose geladen und mindestens 1.5 s beruhigt. Trend: 100-ms-Pose-Schritte, Aufnahme während laufender Bewegung nach ca. 4 s; Timing und Hundertstel können variieren. Kritischer Abstand ist durch depth ≤ 22 auf ca. 0.36 mm begrenzt. Backenöffnung und laterale Auslenkung verändern die tatsächliche Distanz.

Live 17–20: Playwright page.routeWebSocket fängt /ws ab (kein Backend). Der unveränderte poseFrame-Generator liefert mm-Koordinaten; der Mock sendet ca. 30 Hz mit streng steigendem t = Frame/30, confidence 0.99 und calibrated true. Nach 1.5 s Stream wird direkt aufgenommen (17/18) oder das Senden bei offener Verbindung gestoppt (19/20). Die Browser-Uhr wird bei 300/1200 ms seit letzter Nachricht für den Screenshot angehalten. Ghost zeigt keinen aktuellen Abstand; last bezieht sich auf die letzte Messung, die Sekundenanzeige zählt ab Ghost-Beginn.

- 01_far_overview.png: Fern; Scope in Bereitschaft; Quelle debug; Pose {"depth":14.36,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 7.9 mm; Ansicht Overview.
- 02_near_overview.png: Nah; Scope aktiv; Quelle debug; Pose {"depth":18.36,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 3.9 mm; Ansicht Overview.
- 03_warning_overview.png: Warnbereich; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- 04_critical_overview.png: Kritischer Abstand (Tiefenlimit 22 mm); Quelle debug; Pose {"depth":22,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 0.3 mm; Ansicht Overview.
- 05_warning_surgeon.png: Warnbereich aus Operateursperspektive; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Surgeon.
- 06_warning_sagittal.png: Warnbereich im Sagittalschnitt; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Sagittal.
- 07_warning_fundus.png: Warnbereich in Fundusansicht; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Fundus.
- 08_approach_trend.png: Annäherung über 4 s von ca. 5 auf 1.5 mm; Skript: depth startet bei 17.36 mm, +0.0875 mm alle 100 ms (0.875 mm/s), Zielaufnahme ab Schritt 40 (depth 20.86 mm); folgende Pose ist nur das Aufnahmefenster; Quelle debug; Pose {"depth":"20.86 → 21.822499999999998","u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.5 → 1.3 (Messungen vor/nach Screenshot; siehe PNG) mm; Ansicht Overview.
- 09_nodata_live.png: Live tracking ohne Messdaten; Quelle live; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand — mm; Ansicht Overview.
- 10_display_menu_open.png: Display-Menü geöffnet; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- 11_legend_and_nav.png: Legende und Navigation; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- 12_jaws_closed.png: Backen geschlossen; Quelle debug; Pose {"depth":18.36,"u":0,"v":0,"half":0.02,"roll":0}; angezeigter Abstand 3.9 mm; Ansicht Overview.
- 13_jaws_wide.png: Backen maximal geöffnet; Quelle debug; Pose {"depth":18.36,"u":0,"v":0,"half":0.4,"roll":0}; angezeigter Abstand 4.0 mm; Ansicht Overview.
- 14_lateral_sway.png: Seitliche Auslenkung; Schatten/Landepunkt versetzt; Quelle debug; Pose {"depth":17.42,"u":0.5,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 2.0 mm; Ansicht Overview.
- 15_asymmetric_near.png: Asymmetrisch: L näher (≈1.3 vs 1.9 mm), beide Near retina; Quelle debug; Pose {"depth":18,"u":0.5,"v":0,"half":0.3,"roll":0}; angezeigter Abstand 1.3 mm; Ansicht Overview.
- 16_asymmetric_critical.png: Asymmetrisch: L kritisch, R Near retina; Quelle debug; Pose {"depth":19.15,"u":0.5,"v":0,"half":0.3,"roll":0}; angezeigter Abstand 0.4 mm; Ansicht Overview.
- 17_live_critical.png: Live: stetiger kritischer Rahmen und Live-Genauigkeitshinweis; Quelle live; Pose {"depth":22,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 0.3 mm; Ansicht Overview; 30-Hz-WebSocket-Mock, 46 Frames, Status ■ Critical proximity, Hinweis Sim: Critical shown → tip 1–5.6 mm away.
- 18_live_near.png: Live: Near retina; Quelle live; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview; 30-Hz-WebSocket-Mock, 45 Frames, Status ▲ Near retina, Hinweis Sim: Near shown → tip 1.6–3.4 mm away.
- 19_live_held.png: Live: Signal gap, letzter Wert gehalten (300 ms); Quelle live; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview; Messalter 300 ms, Browser-Uhr für PNG angehalten; 30-Hz-WebSocket-Mock, 45 Frames, Status ▲ Near retina, Hinweis Signal gap · holding last value.
- 20_live_ghost_last.png: Live: Ghost mit letzter kritischer Messung (1200 ms); Quelle live; Pose {"depth":22,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand — mm; Ansicht Overview; Messalter 1200 ms, Browser-Uhr für PNG angehalten; 30-Hz-WebSocket-Mock, 41 Frames, Status ○ No measurement · last: Critical proximity 0.3 mm, 0 s ago, Hinweis Distances unavailable until a complete frame arrives.
- crop_approach.png: Warnbereich, Panel-Ausschnitt; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- crop_distance.png: Warnbereich, Panel-Ausschnitt; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- crop_instrument.png: Warnbereich, Panel-Ausschnitt; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.
- crop_scope.png: Warnbereich, Panel-Ausschnitt; Quelle debug; Pose {"depth":21.16,"u":0,"v":0,"half":0.1,"roll":0}; angezeigter Abstand 1.1 mm; Ansicht Overview.

JavaScript-Laufzeitfehler: keine.
