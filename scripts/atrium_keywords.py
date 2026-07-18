#!/usr/bin/env python3
"""Zero-dependency client for the ATRIUM LLM Enrichment (keyword) API.

Uploads document text lines (TXT/CSV/TEITOK file, or plain lines on stdin) to
a running instance of the FastAPI service in `service/api.py` and returns
vocabulary-guided archaeological keywords per line: Czech + English terms, a
TEATER/AMCR category, and a confidence score (local server by default, remote
via --base-url or the ATRIUM_LE_URL env variable).

Only the Python 3 standard library is used - no pip installs required.

Usage:
    python3 scripts/atrium_keywords.py lines.txt
    python3 scripts/atrium_keywords.py lines.csv --backend ollama --top-k 5
    python3 scripts/atrium_keywords.py page.teitok.xml --format json
    python3 scripts/atrium_keywords.py - --doc-id CTX1 < lines.txt
    python3 scripts/atrium_keywords.py --info

Exit codes:
    0 - success
    1 - client-side error (bad arguments, unreadable file)
    2 - server unreachable (connection refused / timeout)
    3 - server-side error (HTTP 4xx/5xx)
"""

import argparse
import csv
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("ATRIUM_LE_URL", "http://localhost:8000")
INPUT_SUFFIXES = (".txt", ".csv", ".xml")
MAX_UPLOAD_MB = 5  # mirrors the server's MAX_UPLOAD_MB default
BACKENDS = ("openrouter", "ollama", "local")
RETRY_STATUS = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 10


def build_multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Encode form fields and one file as multipart/form-data using only the stdlib."""
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    lines.append(f"--{boundary}".encode())
    lines.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"'.encode())
    lines.append(f"Content-Type: {mime}".encode())
    lines.append(b"")
    lines.append(file_path.read_bytes())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_json(url: str, data: bytes = None, content_type: str = None, timeout: int = 1800) -> dict:
    """POST (or GET when data is None) and decode a JSON response, with retry on 502/503/504.

    The long default timeout is deliberate: LLM extraction is the slowest call
    in the ATRIUM family (minutes for a full document).
    """
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in RETRY_STATUS and attempt < RETRY_ATTEMPTS:
                print(
                    f"[retry {attempt}/{RETRY_ATTEMPTS}] HTTP {e.code}, waiting {RETRY_WAIT_S}s...",
                    file=sys.stderr,
                )
                time.sleep(RETRY_WAIT_S)
                last_error = f"HTTP {e.code}: {detail}"
                continue
            if e.code == 429:
                print(
                    "Server busy (HTTP 429): the concurrent-extraction limit is reached. Retry later.",
                    file=sys.stderr,
                )
            elif e.code == 501:
                print(
                    "backend=local is CLI-only on the server (HTTP 501) - use --backend openrouter "
                    "or --backend ollama, or run llm_run.py directly.",
                    file=sys.stderr,
                )
            else:
                print(f"Server error - HTTP {e.code}: {detail}", file=sys.stderr)
            sys.exit(3)
        except (urllib.error.URLError, TimeoutError) as e:
            print(
                f"Cannot reach the API at {url} ({e}).\nIs the server running? Start it with: bash scripts/server.sh",
                file=sys.stderr,
            )
            sys.exit(2)
    print(f"Server error after {RETRY_ATTEMPTS} attempts - {last_error}", file=sys.stderr)
    sys.exit(3)


def extract_file(base_url: str, path: Path, backend: str, top_k: int) -> dict:
    """Upload one file to POST /extract_keywords."""
    if not path.name.lower().endswith(INPUT_SUFFIXES):
        print(f"Skipping {path}: unsupported type. Allowed: .txt, .csv, .teitok.xml", file=sys.stderr)
        return {}
    size = path.stat().st_size
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        print(
            f"Skipping {path}: {size} bytes exceeds the {MAX_UPLOAD_MB} MB server upload limit - "
            "split the document first",
            file=sys.stderr,
        )
        return {}
    fields = {"top_k": top_k}
    if backend:
        fields["backend"] = backend
    body, content_type = build_multipart(fields, file_field="file", file_path=path)
    return http_json(f"{base_url}/extract_keywords", data=body, content_type=content_type)


def extract_stdin(base_url: str, doc_id: str, backend: str, top_k: int) -> dict:
    """Read plain text lines from stdin and send them to POST /extract_keywords_text."""
    lines = [line.strip() for line in sys.stdin.read().splitlines() if line.strip()]
    if not lines:
        print("No text lines on stdin.", file=sys.stderr)
        sys.exit(1)
    payload = {"doc_id": doc_id, "lines": lines, "top_k": top_k}
    if backend:
        payload["backend"] = backend
    return http_json(
        f"{base_url}/extract_keywords_text",
        data=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )


def result_rows(name: str, envelope: dict) -> list[tuple]:
    """Flatten an envelope into (doc, page, line, category, confidence, kw_cs, kw_en) rows."""
    rows = []
    for line in envelope.get("lines", []):
        rows.append(
            (
                envelope.get("doc_id", name),
                line.get("page"),
                line.get("line"),
                line.get("category", ""),
                line.get("confidence"),
                "; ".join(line.get("keywords_cs") or []),
                "; ".join(line.get("keywords_en") or []),
            )
        )
    return rows


def print_table(rows: list[tuple], as_csv: bool) -> None:
    header = ("DOC", "PAGE", "LINE", "CATEGORY", "CONF", "KEYWORDS_CS", "KEYWORDS_EN")
    if as_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        for row in rows:
            conf = "" if row[4] is None else f"{row[4]:.2f}"
            writer.writerow([row[0], row[1], row[2], row[3], conf, row[5], row[6]])
    else:
        print(f"{header[0]:<20} {header[1]:>4} {header[2]:>4} {header[3]:<24} {header[4]:>5} {header[5]}")
        for row in rows:
            conf = "    -" if row[4] is None else f"{row[4]:>5.2f}"
            keywords = row[5] if len(row[5]) <= 45 else row[5][:42] + "..."
            print(f"{row[0]:<20} {str(row[1]):>4} {str(row[2]):>4} {row[3]:<24} {conf} {keywords}")


def summarize(envelope: dict) -> None:
    stats = envelope.get("stats") or {}
    print(
        f"# doc_id={envelope.get('doc_id')} backend={envelope.get('backend')} "
        f"model={envelope.get('model')} processed={stats.get('processed')} "
        f"filtered={stats.get('skipped_filter')} errors={stats.get('skipped_error')}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="TXT/CSV/TEITOK file(s) to enrich, or '-' for stdin lines")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL}, env: ATRIUM_LE_URL)"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="LLM backend (default: server-side default; 'local' is CLI-only and answers 501)",
    )
    parser.add_argument("--top-k", type=int, default=10, help="max keywords per line, 1-50 (default: 10)")
    parser.add_argument("--doc-id", default="document", help="document id for stdin input (default: document)")
    parser.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="output format (default: table)"
    )
    parser.add_argument("--info", action="store_true", help="print service capabilities and limits, then exit")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.info:
        print(json.dumps(http_json(f"{base_url}/info", timeout=60), indent=2))
        return

    if not args.files:
        parser.error("no input files given (or use --info)")

    envelopes = {}
    rows = []
    for name in args.files:
        if name == "-":
            envelope = extract_stdin(base_url, args.doc_id, args.backend, args.top_k)
        else:
            path = Path(name)
            if not path.is_file():
                print(f"File not found: {path}", file=sys.stderr)
                sys.exit(1)
            envelope = extract_file(base_url, path, args.backend, args.top_k)
        if not envelope:
            continue
        envelopes[name] = envelope
        summarize(envelope)
        rows.extend(result_rows(name, envelope))

    if not envelopes:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(envelopes if len(envelopes) > 1 else next(iter(envelopes.values())), indent=2, ensure_ascii=False))
    else:
        if not rows:
            print("No lines passed the quality filter. Use --format json for the full envelope.", file=sys.stderr)
            return
        print_table(rows, as_csv=(args.format == "csv"))


if __name__ == "__main__":
    main()
