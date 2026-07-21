import { useState } from "react";
import type { FleetConfig } from "../../types";
import { probeConnection, type ConnectionProbe } from "../../api";

interface Props {
  config: FleetConfig;
  onChange: (next: FleetConfig) => void;
  lastProbe?: ConnectionProbe | null;
  onProbed?: (probe: ConnectionProbe) => void;
}

const PRESETS = [
  {
    label: "Railway quant",
    apiBase: "https://quant-production-5533.up.railway.app",
  },
  {
    label: "Local webhook",
    apiBase: "http://127.0.0.1:8000",
  },
];

export function SettingsDrawer({ config, onChange, lastProbe, onProbed }: Props) {
  const [probing, setProbing] = useState(false);
  const [probe, setProbe] = useState<ConnectionProbe | null>(lastProbe || null);

  const updateBot = (idx: number, patch: Partial<FleetConfig["bots"][number]>) => {
    const bots = config.bots.map((b, i) => (i === idx ? { ...b, ...patch } : b));
    onChange({ ...config, bots });
  };

  const runProbe = async () => {
    setProbing(true);
    try {
      const result = await probeConnection(config);
      setProbe(result);
      onProbed?.(result);
    } finally {
      setProbing(false);
    }
  };

  return (
    <div className="space-y-5 text-[12px]">
      <section className="space-y-3 border border-[var(--line)] p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">
            Desktop connection
          </h3>
          <button
            onClick={() => void runProbe()}
            disabled={probing}
            className="border border-[var(--line)] px-2 py-1 text-[11px] tracking-wide text-[var(--muted)] hover:text-[var(--text)] disabled:opacity-50"
          >
            {probing ? "Testing…" : "Test connection"}
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.apiBase}
              className="border border-[var(--line)] px-2 py-1 text-[11px] text-[var(--muted)] hover:text-[var(--text)]"
              onClick={() => onChange({ ...config, apiBase: p.apiBase })}
            >
              {p.label}
            </button>
          ))}
        </div>
        {probe && (
          <div className="space-y-2 text-[11px]">
            <p>
              Mode:{" "}
              <span style={{ color: probe.mode === "offline" ? "var(--down)" : "var(--live)" }}>
                {probe.mode}
              </span>
            </p>
            <p className="text-[var(--muted)]">{probe.fleetApiDetail}</p>
            <ul className="space-y-1 border-t border-[var(--line)] pt-2">
              {probe.healthHits.map((h) => (
                <li key={h.id} className="flex justify-between gap-2">
                  <span>{h.display_name}</span>
                  <span style={{ color: h.ok ? "var(--live)" : "var(--down)" }}>
                    {h.status}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <p className="text-[11px] text-[var(--muted)] leading-relaxed">
          API base should be the quant dashboard host. Leave token empty unless the
          server sets <code>FLEET_REQUIRE_AUTH=1</code>.
        </p>
      </section>

      <label className="block space-y-1">
        <span className="text-[var(--muted)]">Fleet API base URL</span>
        <input
          className="w-full border border-[var(--line)] bg-black/30 px-2 py-2 text-[var(--text)] outline-none focus:border-[var(--accent)]"
          value={config.apiBase}
          placeholder="https://quant-production-5533.up.railway.app"
          onChange={(e) => onChange({ ...config, apiBase: e.target.value })}
        />
      </label>
      <label className="block space-y-1">
        <span className="text-[var(--muted)]">Read token</span>
        <input
          type="password"
          className="w-full border border-[var(--line)] bg-black/30 px-2 py-2 text-[var(--text)] outline-none focus:border-[var(--accent)]"
          value={config.token}
          onChange={(e) => onChange({ ...config, token: e.target.value })}
        />
      </label>
      <div className="grid grid-cols-2 gap-3">
        <label className="block space-y-1">
          <span className="text-[var(--muted)]">Health poll (ms)</span>
          <input
            type="number"
            className="w-full border border-[var(--line)] bg-black/30 px-2 py-2"
            value={config.healthPollMs}
            onChange={(e) => onChange({ ...config, healthPollMs: Number(e.target.value) || 10000 })}
          />
        </label>
        <label className="block space-y-1">
          <span className="text-[var(--muted)]">Curve poll (ms)</span>
          <input
            type="number"
            className="w-full border border-[var(--line)] bg-black/30 px-2 py-2"
            value={config.curvePollMs}
            onChange={(e) => onChange({ ...config, curvePollMs: Number(e.target.value) || 45000 })}
          />
        </label>
      </div>

      <div className="space-y-3">
        <h3 className="text-[11px] uppercase tracking-[0.14em] text-[var(--muted)]">Bot registry</h3>
        {config.bots.map((bot, idx) => (
          <div key={bot.id} className="space-y-2 border border-[var(--line)] p-3">
            <div className="flex items-center justify-between">
              <input
                className="bg-transparent text-[var(--text)] outline-none"
                value={bot.display_name}
                onChange={(e) => updateBot(idx, { display_name: e.target.value })}
              />
              <label className="flex items-center gap-2 text-[var(--muted)]">
                <input
                  type="checkbox"
                  checked={bot.enabled}
                  onChange={(e) => updateBot(idx, { enabled: e.target.checked })}
                />
                on
              </label>
            </div>
            <input
              className="w-full border border-[var(--line)] bg-black/20 px-2 py-1 text-[11px]"
              value={bot.strategy_instance}
              onChange={(e) => updateBot(idx, { strategy_instance: e.target.value })}
              placeholder="strategy_instance"
            />
            <input
              className="w-full border border-[var(--line)] bg-black/20 px-2 py-1 text-[11px]"
              value={bot.health_url}
              onChange={(e) => updateBot(idx, { health_url: e.target.value })}
              placeholder="health URL"
            />
            <input
              className="w-full border border-[var(--line)] bg-black/20 px-2 py-1 text-[11px]"
              value={bot.color}
              onChange={(e) => updateBot(idx, { color: e.target.value })}
              placeholder="#color"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
