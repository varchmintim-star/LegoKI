"""Die LEGO-KI-Präsentation lokal mit Kameraunterstützung bereitstellen."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port muss zwischen 1 und 65535 liegen.")
    folder = Path(__file__).resolve().parent / "presentation/dist"
    if not (folder / "index.html").exists():
        parser.error("Präsentationsdateien fehlen.")
    mimetypes.add_type("application/wasm", ".wasm")
    handler = partial(SimpleHTTPRequestHandler, directory=str(folder))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Präsentation: http://localhost:{args.port}", flush=True)
        print("Im Browser öffnen. Beenden mit Strg+C.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
