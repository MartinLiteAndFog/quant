import { useEffect, useRef, useState } from "react";
import { CURATED_GENE_MODULES, STRATEGY_RECIPES, GENE_LIBRARY_LIMITS, addGeneEntry, loadGeneLibrary, parseGeneLibrary,
  saveGeneLibrary, type GeneCandidate, type GeneEntry, type GeneLibraryStore } from "./geneLibrary.ts";
import "./geneLibrary.css";

type Props = { open: boolean; onClose: () => void; candidate?: GeneCandidate };
type Draft = { id?: string; createdAt?: string; name: string; source: string; parameters: string; notes: string; snippet: string };
const emptyDraft = (): Draft => ({ name: "", source: "Eigene Notiz", parameters: "{}", notes: "", snippet: "" });
const toDraft = (entry: GeneEntry): Draft => ({ ...entry, parameters: JSON.stringify(entry.parameters, null, 2) });

export function GeneLibrary({ open, onClose, candidate }: Props) {
  const dialog = useRef<HTMLDialogElement>(null), fileInput = useRef<HTMLInputElement>(null);
  const [library, setLibrary] = useState<GeneLibraryStore>({ schemaVersion: 1, entries: [] });
  const [view, setView] = useState<"sources" | "saved" | "edit">("sources");
  const [selected, setSelected] = useState(CURATED_GENE_MODULES[0]!.id);
  const [draft, setDraft] = useState<Draft>(emptyDraft), [query, setQuery] = useState("");
  const [message, setMessage] = useState(""), [loadError, setLoadError] = useState<string | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const element = dialog.current;
    if (open && element && !element.open) {
      try {
        const result = loadGeneLibrary(window.localStorage); setLibrary(result.library); setLoadError(result.error);
      } catch { setLoadError("Lokaler Bibliotheksspeicher ist in diesem Browser nicht verfügbar."); }
      setError(""); setMessage(""); element.showModal();
    } else if (!open && element?.open) element.close();
  }, [open]);
  const commit = (next: GeneLibraryStore) => {
    if (loadError) throw new Error("Der vorhandene Speicher bleibt erhalten. Exportiere deine Entwürfe; die gespeicherte Bibliothek muss zuerst wiederhergestellt werden.");
    saveGeneLibrary(window.localStorage, next); setLibrary(next); setError("");
  };
  const beginCandidate = () => {
    if (!candidate) return;
    setDraft({ ...emptyDraft(), ...candidate, parameters: JSON.stringify(candidate.parameters, null, 2) });
    setView("edit"); setMessage(""); setError("");
  };
  const submit = () => {
    try {
      const entry: GeneEntry = { id: draft.id ?? `gene-${crypto.randomUUID()}`, createdAt: draft.createdAt ?? new Date().toISOString(),
        name: draft.name.trim(), source: draft.source.trim(), parameters: JSON.parse(draft.parameters), notes: draft.notes, snippet: draft.snippet };
      const next = draft.id ? parseGeneLibrary(JSON.stringify({ schemaVersion: 1, entries: library.entries.map((item) => item.id === draft.id ? entry : item) }))
        : addGeneEntry(library, entry);
      commit(next); setDraft(toDraft(entry)); setMessage("In deiner lokalen Genbibliothek gespeichert.");
    } catch (issue) { setError(issue instanceof Error ? issue.message : "Speichern fehlgeschlagen."); setMessage(""); }
  };
  const exportLibrary = () => {
    try {
      const content = view === "edit" ? { schemaVersion: 1, draft } : library;
      const href = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], { type: "application/json" }));
      const link = document.createElement("a"); link.href = href; link.download = view === "edit" ? "Cloudgarden-Genentwurf.json" : "Cloudgarden-Genbibliothek.json";
      link.click(); setTimeout(() => URL.revokeObjectURL(href), 1000);
      setMessage(view === "edit" ? "Entwurf als Datei exportiert." : "Genbibliothek als Datei exportiert.");
    } catch { setError("Export nicht möglich. Dein Bibliotheksspeicher bleibt erhalten."); }
  };
  const visibleSources = CURATED_GENE_MODULES.filter((item) => `${item.name} ${item.category} ${item.source}`.toLowerCase().includes(query.toLowerCase()));
  const visibleSaved = library.entries.filter((item) => `${item.name} ${item.source} ${item.notes}`.toLowerCase().includes(query.toLowerCase()));
  const module = CURATED_GENE_MODULES.find((item) => item.id === selected) ?? CURATED_GENE_MODULES[0]!;
  return <dialog className="gene-library" ref={dialog} aria-labelledby="gene-library-title" onCancel={(event) => { event.preventDefault(); onClose(); }}>
    <header className="gene-library-header"><div><h2 id="gene-library-title">Genbibliothek</h2><p>Strategiebausteine sammeln, verstehen und wiederfinden.</p></div>
      <button type="button" className="gene-close" onClick={onClose} aria-label="Genbibliothek schließen"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="m6 6 12 12M18 6 6 18" /></svg></button>
    </header>
    <nav className="gene-library-tabs" aria-label="Genbibliothek Ansichten">
      <button type="button" aria-current={view === "sources" ? "page" : undefined} onClick={() => setView("sources")}>Einzelgene <span>{CURATED_GENE_MODULES.length}</span></button>
      <button type="button" aria-current={view === "saved" ? "page" : undefined} onClick={() => setView("saved")}>Meine Sammlung <span>{library.entries.length}</span></button>
      <button type="button" aria-current={view === "edit" ? "page" : undefined} onClick={() => { if (view !== "edit") setDraft(emptyDraft()); setView("edit"); setMessage(""); }}>Eigener Baustein</button>
    </nav>
    <div className="gene-library-content">
      {loadError && <p role="alert" className="gene-error">{loadError} Die gespeicherten Daten wurden nicht überschrieben.</p>}
      {view !== "edit" && <div className="gene-library-search"><label htmlFor="gene-search">Bibliothek durchsuchen</label><input id="gene-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Name, Quelle oder Notiz" /></div>}
      {view === "sources" && <><p>Gene sind einzelne Funktionen. Strategien kombinieren Gene. Eigene Texte sind ungeprüft und werden niemals ausgeführt.</p><details><summary>Strategierezepte aus Einzelgenen</summary>{STRATEGY_RECIPES.map(recipe=><article key={recipe.id}><h3>{recipe.name}</h3><p>{recipe.nodes.map(node=>CURATED_GENE_MODULES.find(g=>g.id===node.geneId)?.name??node.geneId).join(" + ")}</p><small>{recipe.execution==="compiled_profile"?"Implementierte Vorlage; vollständige Parameter unter Pflanzen-Handel.":"Entwurf: noch keine frei ausführbare Strategie."}</small></article>)}</details><div className="gene-library-browser"><div className="gene-module-list" aria-label="Quellmodule">
        {visibleSources.map((item) => <button type="button" key={item.id} aria-pressed={selected === item.id} onClick={() => setSelected(item.id)}><span>{item.category}</span><strong>{item.name}</strong></button>)}
        {!visibleSources.length && <p>Kein passendes Quellmodul. Probiere einen kürzeren Suchbegriff.</p>}
      </div><article className="gene-module-detail"><h3>{module.name}</h3><p>{module.description}</p>
        <dl className="gene-parameters">{Object.entries(module.parameters).filter(([, value]) => typeof value !== "object").map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl>
        <h4>Zeitliche Verfügbarkeit</h4><dl className="gene-parameters">{Object.entries(module.causality).map(([key,value])=><div key={key}><dt>{({inputs:"Datenbedarf",lookback:"Historischer Rückblick",confirmationLag:"Bestätigungsverzögerung",observableAt:"Signal verfügbar",executionAt:"Früheste Ausführung",retrospectivePlot:"Rückdatierte Darstellung",verification:"Prüfstatus",futureRequired:"Zukunftszugriff"} as Record<string,string>)[key]}</dt><dd>{key==="futureRequired"?"verboten":value==="tested"?"Modul mit Kausalitätstests":value==="declarative"?"Deklaration; kein ausführbarer Graph":String(value)}</dd></div>)}</dl><h4>Baustein</h4><pre>{module.snippet}</pre><p className="gene-caveat">{module.caveat}</p>
        <h4>Herkunft</h4><p className="gene-source">{module.source}</p><p className="gene-license">{module.license}</p>
        <button type="button" className="gene-primary" onClick={() => { setDraft({ ...emptyDraft(), name: module.name, source: module.source, parameters: JSON.stringify(module.parameters, null, 2), snippet: module.snippet, notes: `${module.caveat}\n\n${module.license}` }); setView("edit"); setMessage(""); }}>Als eigenen Baustein übernehmen</button>
      </article></div></>}
      {view === "saved" && <div className="gene-saved-list">{visibleSaved.map((entry) => <button key={entry.id} type="button" onClick={() => { setDraft(toDraft(entry)); setView("edit"); setMessage(""); }}><div><strong>{entry.name}</strong><p>{entry.source}</p></div><time dateTime={entry.createdAt}>{new Date(entry.createdAt).toLocaleDateString("de-DE")}</time></button>)}
        {!library.entries.length && <div className="gene-empty"><h3>Hier wächst deine Sammlung.</h3><p>Übernimm ein Quellmodul, sichere die Gene deiner ausgewählten Pflanze oder notiere eine eigene Strategieidee.</p><button type="button" onClick={() => setView("sources")}>Quellmodule ansehen</button></div>}
        {library.entries.length > 0 && !visibleSaved.length && <p>Kein gespeicherter Baustein passt zu dieser Suche.</p>}
      </div>}
      {view === "edit" && <form className="gene-editor" onSubmit={(event) => { event.preventDefault(); submit(); }}>
        <div className="gene-editor-heading"><h3>{draft.id ? "Baustein bearbeiten" : "Neuer Baustein"}</h3><span>Entwurf · wird erst mit Speichern übernommen</span></div>
        <div className="gene-field-pair"><label>Name<input required value={draft.name} maxLength={GENE_LIBRARY_LIMITS.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} /></label><label>Quelle<input required value={draft.source} maxLength={GENE_LIBRARY_LIMITS.source} onChange={(e) => setDraft({ ...draft, source: e.target.value })} /></label></div>
        <label>Notizen<textarea rows={3} value={draft.notes} maxLength={GENE_LIBRARY_LIMITS.notes} onChange={(e) => setDraft({ ...draft, notes: e.target.value })} placeholder="Was macht diesen Baustein aus?" /></label>
        <div className="gene-field-pair"><label>Parameter als JSON<textarea className="gene-code-field" rows={9} value={draft.parameters} maxLength={24000} onChange={(e) => setDraft({ ...draft, parameters: e.target.value })} spellCheck={false} /></label><label>Snippet oder Strategieidee<textarea className="gene-code-field" rows={9} value={draft.snippet} maxLength={GENE_LIBRARY_LIMITS.snippet} onChange={(e) => setDraft({ ...draft, snippet: e.target.value })} spellCheck={false} /></label></div>
        <p className="gene-caveat">Gespeicherter Text wird dokumentiert und niemals ausgeführt. Eine Übernahme in die Simulation braucht eine geprüfte Strategieimplementierung.</p>
        <div className="gene-editor-actions"><button type="submit" className="gene-primary" disabled={Boolean(loadError)}>Baustein speichern</button>{draft.id && <button type="button" onClick={() => { try { commit({ schemaVersion: 1, entries: library.entries.filter((entry) => entry.id !== draft.id) }); setDraft(emptyDraft()); setView("saved"); setMessage("Baustein aus deiner Sammlung entfernt."); } catch (issue) { setError(issue instanceof Error ? issue.message : "Löschen fehlgeschlagen."); } }}>Aus Sammlung entfernen</button>}</div>
      </form>}
      {error && <p className="gene-error" role="alert">{error}</p>}<p className="gene-message" role="status">{message}</p>
    </div>
    <footer className="gene-library-footer"><p>Lokal in diesem Browser · {library.entries.length} / {GENE_LIBRARY_LIMITS.entries} eigene Bausteine</p><div>
      {candidate && <button type="button" className="gene-primary" onClick={beginCandidate}>Ausgewählte Pflanze sammeln</button>}
      <button type="button" onClick={exportLibrary}>{view === "edit" ? "Entwurf exportieren" : "Sammlung exportieren"}</button>
      <button type="button" onClick={() => fileInput.current?.click()}>Sammlung importieren</button>
      <input ref={fileInput} type="file" accept="application/json,.json" hidden aria-label="Genbibliothek importieren" onChange={async (event) => {
        const file = event.target.files?.[0]; event.target.value = ""; if (!file) return;
        try {
          if (file.size > GENE_LIBRARY_LIMITS.bytes) throw new Error("Importdatei ist zu groß.");
          const imported = parseGeneLibrary(await file.text()), ids = new Set(library.entries.map((entry) => entry.id));
          const next = parseGeneLibrary(JSON.stringify({ schemaVersion: 1, entries: [...library.entries, ...imported.entries.filter((entry) => !ids.has(entry.id))] }));
          commit(next); setView("saved"); setMessage(`${next.entries.length - library.entries.length} neue Bausteine importiert.`);
        } catch (issue) { setError(issue instanceof Error ? issue.message : "Import fehlgeschlagen."); }
      }} />
    </div></footer>
  </dialog>;
}
