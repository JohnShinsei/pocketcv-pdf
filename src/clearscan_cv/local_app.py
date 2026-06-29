from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _pick_port(host: str, requested_port: int) -> int:
    for port in range(requested_port, requested_port + 30):
        if _port_is_free(host, port):
            return port
    raise RuntimeError(f"No free local port found from {requested_port} to {requested_port + 29}.")


def _open_browser_later(url: str) -> None:
    def run() -> None:
        time.sleep(1.0)
        webbrowser.open(url)

    threading.Thread(target=run, daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the PocketCV PDF local backend app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Keep 127.0.0.1 for local-only processing.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred local port.")
    parser.add_argument("--no-browser", action="store_true", help="Start the backend without opening a browser.")
    parser.add_argument("--api-docs", action="store_true", help="Open FastAPI docs instead of the local app page.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install local backend dependencies with: pip install -e .[api]") from exc

    port = _pick_port(args.host, args.port)
    path = "/docs" if args.api_docs else "/local"
    url = f"http://{args.host}:{port}{path}"
    print("PocketCV PDF local backend")
    print(f"URL: {url}")
    print("Images are processed on this machine via FastAPI/OpenCV.")
    if not args.no_browser:
        _open_browser_later(url)
    uvicorn.run("clearscan_cv.api:app", host=args.host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
