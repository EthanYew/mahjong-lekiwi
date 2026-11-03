"""Go through a bench session's pickups and returns and say which tiles really moved, against the scans.

    python scripts/bump_report.py                           # the latest bench session, opens in your browser
    python scripts/bump_report.py sessions/<folder>
    python scripts/bump_report.py --no-serve                # only write <session>/bump_report.html

`bench.py --unattended` writes the page when it ends; this serves it on your own machine so its buttons can save. Each
verdict goes straight into the session's events.jsonl, and session_report.py and status.py follow it. Ctrl-C when done.
Run from the main checkout (sessions are per checkout). What's on the page: mahjong/bump_report.py.
"""
import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import bump_report, config


def latest_bench_session():
    sessions = sorted(p for p in config.SESSIONS.glob("*_bench") if (p / "events.jsonl").exists())
    if not sessions:
        raise SystemExit(f"no bench sessions in {config.SESSIONS}")
    return sessions[-1]


def make_server(session_dir, port=0):
    """A server for one session's page, verdicts and pictures, on this machine only. `port` 0: any free one."""
    root = session_dir.resolve()
    events, found = bump_report.load(session_dir)
    (session_dir / bump_report.PAGE).write_text(bump_report.render(session_dir.name, found, bump_report.counts(events, found)))
    types = {".jpg": "image/jpeg", ".html": "text/html; charset=utf-8"}

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0].lstrip("/")
            if not path:
                events = bump_report.read_events(session_dir)
                page = bump_report.render(session_dir.name, found, bump_report.counts(events, found))
                return self.send(200, page.encode(), types[".html"])
            file = (root / path).resolve()
            if not file.is_relative_to(root) or file.suffix not in types or not file.is_file():
                return self.send(404, b"not found")
            self.send(200, file.read_bytes(), types[file.suffix])

        def do_POST(self):
            # JSON only: a browser won't send that from another site's page without asking first, and nothing here says yes
            if self.path != "/verdict" or self.headers.get("Content-Type") != "application/json":
                return self.send(404, b"not found")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                bump_report.record_verdict(session_dir, found, body.get("id"), body.get("verdict"))
            except ValueError as e:
                return self.send(400, str(e).encode())
            c = bump_report.counts(bump_report.read_events(session_dir), found)
            self.send(200, json.dumps(c).encode(), "application/json")

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("session", nargs="?", type=Path, default=None, help="session folder (default: the latest bench session)")
    p.add_argument("--no-serve", action="store_true", help="only write the page, readable with no buttons working")
    p.add_argument("--port", type=int, default=0, help="port for the page (default: any free one)")
    args = p.parse_args(argv)
    session_dir = args.session or latest_bench_session()
    if not (session_dir / "events.jsonl").exists():
        raise SystemExit(f"{session_dir} has no events.jsonl")
    if args.no_serve:
        c = bump_report.write(session_dir)
        print(f"{session_dir / bump_report.PAGE}: {c['flagged']} self-reported bump(s), {c['checked']} checked")
        return
    server = make_server(session_dir, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"{session_dir}\n{url}  (verdicts save as you click; Ctrl-C when done)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
