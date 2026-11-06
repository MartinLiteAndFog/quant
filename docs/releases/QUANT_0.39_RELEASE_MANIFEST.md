# QUANT 0.39 Offline-Release — Prüfsummen und Übergabe

**Datum:** 2026-11-06  
**Status:** Forschungsprototyp / lokale Offline-App; kein Railway-Deployment

| Artefakt | SHA-256 |
|---|---|
| `QUANT_IMBA_Dashboard_0.39_Research_Worker_Cache_Offline.html` | `ba8be0336c28025a72cfd92ac9a64afbc34c5e181203895aaa2e92d90fb0e6aa` |
| `QUANT_IMBA_Dashboard_0.39_Research_Worker_Cache_Offline.zip` | `5d7fe9fdaeb1266b628c2a0d0426a88e98a426915aab6c9db474742a4cb3412e` |

ZIP enthält die eigenständig öffnende, inline vollständig eingebettete Offline-HTML-App, den Worker als separat inspizierbaren JS-Quelltext, die Bridge, UI-Funktionen, den 0.38-Portfolio-Kern und README.

**GitHub-Status:** Dieses Manifest und die detaillierte Architektur unter `docs/research/quant_dashboard_0_39_worker_ranking_cache_2026-11-06.md` wurden committed. Die großen HTML-/ZIP-Release-Binärdaten sind **nicht** ins GitHub-Repository oder GitHub Releases übertragen. Sie stehen in der QUANT-Chat-Konversation als verifizierte Downloads zur Verfügung.

**QA:** Node Syntaxprüfung der drei neuen JS-Dateien, ZIP CRC geprüft. Headless Chromium mit 28.800 SOL-1m-Kerzen, 4 Bots, 16 Portfolio-Varianten, nochmaligem Sweep (komplette Wiederverwendung aus Cache), QRC gzip Export/Import (26 Elemente) und Tagesarchivumschaltung ohne JS-Fehler. Der Test lief in einer opaque Browser-Origin: IndexedDB-Persistenz zwischen vollständigen Browserneustarts ist **noch nicht geprüft**.

**Nächster Meilenstein:** großen Parametergitter-Sweep und Walk-Forward-Engine auf dieselben Signals/Rankings/Portfolio-Cases umstellen und Cachepersistence in stabilem lokalem Browserkontext verifizieren.
