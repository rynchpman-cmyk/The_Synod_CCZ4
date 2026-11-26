"""Local HTTP server for the active-repair dashboard suite."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent
SHOWCASE_ROOT = ROOT / "runs" / "repair_showcases"
SCENARIOS = {
    "interaction_dominant": {
        "title": "Interaction Dominant",
        "subtitle": "Slow target · heavy repair mode · interaction-energy takeover",
        "accent": "#ffad42",
    },
    "ringing_trap": {
        "title": "Ringing Trap",
        "subtitle": "Fast quench · five phase reversals · outward repair shells",
        "accent": "#6ee7ff",
    },
}


def _summary(name: str) -> dict:
    path = SHOWCASE_ROOT / name / "single_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[0] if isinstance(payload, list) else payload


def catalog() -> dict:
    scenarios = []
    for key, presentation in SCENARIOS.items():
        directory = SHOWCASE_ROOT / key
        available = directory.is_dir() and any(directory.glob("*.npz"))
        scenarios.append(
            {
                "id": key,
                **presentation,
                "available": available,
                "summary": _summary(key) if available else None,
            }
        )
    return {"suite": "Causal Repair Observatory", "scenarios": scenarios}


def trajectory(name: str) -> dict:
    if name not in SCENARIOS:
        raise KeyError(name)
    paths = sorted((SHOWCASE_ROOT / name).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no trajectory found for {name}")
    with np.load(paths[0]) as data:
        arrays = {key: data[key].tolist() for key in data.files}
    return {
        "id": name,
        **SCENARIOS[name],
        "summary": _summary(name),
        "arrays": arrays,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    cache: dict[str, bytes] = {}

    def log_message(self, format: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")

    def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value: object, status: int = 200) -> None:
        self._send(
            json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802 - HTTP handler API
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/health":
                self._json({"status": "ok", "showcase_root": str(SHOWCASE_ROOT)})
                return
            if path == "/api/catalog":
                self._json(catalog())
                return
            if path.startswith("/api/showcase/"):
                name = path.rsplit("/", 1)[-1]
                cache_key = f"trajectory:{name}"
                if cache_key not in self.cache:
                    self.cache[cache_key] = json.dumps(
                        trajectory(name), separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                self._send(self.cache[cache_key], "application/json; charset=utf-8")
                return
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            candidate = (STATIC / relative).resolve()
            if STATIC.resolve() not in candidate.parents and candidate != STATIC.resolve():
                self._json({"error": "invalid path"}, 400)
                return
            if not candidate.is_file():
                candidate = STATIC / "index.html"
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
                content_type += "; charset=utf-8"
            self._send(candidate.read_bytes(), content_type)
        except KeyError:
            self._json({"error": "unknown showcase"}, 404)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)
        except Exception as error:  # keep the local UI informative
            self._json({"error": f"{type(error).__name__}: {error}"}, 500)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Causal Repair Observatory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Causal Repair Observatory running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
