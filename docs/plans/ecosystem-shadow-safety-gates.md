# ECO-13 — Read-only Gate-Review vor irgendeinem Shadow-/Live-Cutover

**Status:** Datenprüfmodul auf [ECO-13 Draft PR #58](https://github.com/MartinLiteAndFog/quant/pull/58) gestapelt. **Kein Live-Zugriff, kein TradingView-Alert, keine Registeränderung, kein Deployment, kein neues Order-/Hedge-/Positionskommando.** Es kann mit reinen synthetischen Tests entwickelt werden und benötigt **keine** lokale SOL-Historie.

## Warum dieses Modul nötig ist

`prepare_strategy_snapshot()` aus ECO-13 akzeptiert ein als `eligible_for_sowing` und `beats` **behauptetes** ResearchCandidate mit `out_of_sample=true`. Das ist für einen **prüfbaren Entwurf** sinnvoll, **aber kein unabhängiger statistischer oder Venue-Sicherheitsnachweis**. Insbesondere ECO-12 v1.1 darf keine positive statistische Entscheidung aus mehrfach wiederverwendetem, identischem Marktholdout behaupten.

`src/quant/ecosystem/adapters/shadow_audit.py` stellt deshalb einen **separaten verpflichtenden Prüfbericht** bereit:

- `snapshot_digest(snapshot)`: SHA-256 über die kanonische JSON-Darstellung des konkreten, versionierten Strategie-/Genom-/Kandidaten-Snapshots. Jede Genmutation, Codeversionsänderung oder Revisionsänderung macht ältere Nachweise für das **neue** Artefakt ungültig.
- `GateAttestation(gate_id, status, snapshot_sha256, checked_at, reviewer, artifact_sha256, artifact_ref)`: **Claim** mit explizitem Human-/Artefakt-Referenzfeld, genauer Gateverankerung und UTC-Timestamp. `pass_claimed` benötigt einen separat referenzierten Artefakt-SHA256. Der SHA wird hier **nicht vom Modul neu aus dem Artefakt gelesen** und ist deshalb keine bestätigte Prüfsumme; ein willkürlicher Selbstbericht bleibt unbewiesen.
- `review_shadow_gates(snapshot, attestations, as_of, max_attestation_age_days=90)`: listet grüne Behauptungen, fehlende, gescheiterte und alte oder unklare Gates; Duplikate und Zukunftsdatierungen werden abgelehnt. **Auch 16 von 16 behauptete grüne Gates ergeben immer `cutover="NO_GO"`, `live_routing_authorized=false`, `order_sent=false`, `artifact_hashes_unverified=true` und `independent_market_oos_verified_by_this_tool=false`.** Erst eine später bewusst separat entwickelte, unabhängig autorisierte Operator-/Venue-Prüfung könnte die Nachweise lesen; auch sie benötigt eine ausdrückliche menschliche Entscheidung.

## Die 16 Pflichtgates

| Prüfgate | Extern nachzuweisender Sachverhalt |
|---|---|
| `original_market_data_identity_and_license` | Originalquelle, SHA und erlaubte Nutzung |
| `instrument_and_contract_multiplier` | Exakte Spot-/Futures-/Perp-Spezifikation, Lot/Notional-Einheiten |
| `tradingview_indicator_and_signal_parity` | Originalchart, 5m-Renko Wicks/Offsets und Signale |
| `independent_out_of_sample_and_multiplicity` | Unberührter OOS-Block, Signifikanz über unabhängige Markteinheiten, viele Versuche korrigiert |
| `fee_slippage_spread_and_funding` | Venue-kalibrierte Kosten/Zeitpunkte |
| `causal_net_pnl_and_mark_to_market` | Fills, Netto-TWR, Gebühren, Drawdown und externe Cashflows |
| `hedge_mode_read_only_check` | Read-only Positionsmodus, keine Änderung |
| `single_authoritative_order_writer` | Nur ein autorisierter Orderpfad |
| `reduce_only_tp_sl_owned_by_strategy` | Per-Bot Protective-SL/TP, Hedge-Side, kein versehentliches Opening |
| `no_cross_bot_cancel_all` | Kein fremdes SL/TP beim Cancel-All |
| `minimum_lot_and_step_size` | Derzeitige Instrument-MinQty/Qty-Floors und Rundung |
| `fill_order_and_position_reconciliation` | Tatsächliche Fills, Positionen und Orders abgeglichen |
| `crash_recovery_event_idempotence` | Replay nach Prozessabbruch, kein doppelter Entry/Exit |
| `global_loss_margin_and_90_10_reserve` | Risiko-Obergrenzen und unabhängige 90/10-Family-NAV-Budgetlogik |
| `shadow_only_event_replay` | Separate, nicht ausführende Shadow-Pipeline |
| `separate_operator_cutover_procedure` | Eine **separate** geplante, explizit autorisierte manuelle Betriebsprozedur |

## Statussemantik und Abnahme

- **Keine Attestations:** Alle Gates fehlen → `NO_GO_missing_or_failed_checks`.
- **Eine echte Gegenanzeige oder zu alte Attestation:** `failed` bzw. `unknown` → weiterhin `NO_GO`.
- **Alle 16 mit fremden, von Menschen überprüfbar referenzierten Artefakten versehen:** Nur `claims_complete_needs_independent_review`, **kein GO**.
- **Verändertes Trading-Genom/Codeversion:** Prüfnachweise tragen anderen Snapshot-Digest und werden zurückgewiesen.
- Kein Cloud-/Broker-/GitHub-Call im Modul. Es berechnet **weder** Kapitalfreigabe noch Trading-/Portfolio-Alpha. Synthetische Testclaims sind keine echten Stakeholder-Abnahmen.

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_shadow_audit.py -v
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_shadow_bridge.py -v
```

**Nächste Schritte nur bei manueller Datenfreigabe:** Echte Belege für konkretes Kraken/KuCoin-Instrument, TradingView-Parität, Hedge-Mode-/SL-/Order-Reconciliation, Live-Broker-Risikoregeln und echten OOS-Benchmark in den Bericht übernehmen, dann **außerhalb dieses Research-Moduls** durch eine verantwortliche Person gegen die Originalnachweise prüfen. Kein automatischer Übergang auf live. Änderungen ausschließlich auf isoliertem Forschungsbranch.
