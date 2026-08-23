/** HTTP transport: Tauri native fetch when available (no CORS), else browser fetch. */

export async function fleetFetch(
  url: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers || {});
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  // Fleet cockpit must never show a cached board after Refresh / poll.
  if (!headers.has("Cache-Control")) headers.set("Cache-Control", "no-cache");
  if (!headers.has("Pragma")) headers.set("Pragma", "no-cache");

  const isTauri =
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window);

  if (isTauri) {
    try {
      const { fetch: tauriFetch } = await import("@tauri-apps/plugin-http");
      const res = await tauriFetch(url, {
        method: (init.method as string) || "GET",
        headers: Object.fromEntries(headers.entries()),
      });
      return res as unknown as Response;
    } catch {
      // Fall through to webview fetch
    }
  }

  return fetch(url, { ...init, headers, cache: "no-store" });
}
