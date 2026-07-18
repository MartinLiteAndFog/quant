from __future__ import annotations

import os


def main() -> None:
    process = str(os.getenv("RAILWAY_PROCESS", "web")).strip().lower()
    if process == "bot":
        from quant.execution.railway_bot import main as bot_main

        bot_main()
        return
    if process != "web":
        raise ValueError("RAILWAY_PROCESS must be 'web' or 'bot'")

    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    print(f"starting Railway web process on 0.0.0.0:{port}", flush=True)
    uvicorn.run("quant.execution.webhook_server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
