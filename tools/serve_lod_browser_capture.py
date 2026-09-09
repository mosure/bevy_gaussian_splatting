#!/usr/bin/env python3
"""Read-only local package server with exact ranges, immutable ETags and capture identity."""
import argparse
import hashlib
import http.server
import json
import mimetypes
import pathlib
import re
import subprocess
import threading
import urllib.parse

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "www/out/lod-browser"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPO
    ).decode().split("\0")
    result = hashlib.sha256()
    for name in sorted(set(names)):
        if not name or pathlib.Path(name).suffix not in {".rs", ".wgsl", ".toml", ".lock"}:
            continue
        path = REPO / name
        if path.is_file():
            result.update(name.encode() + b"\0" + bytes.fromhex(digest(path)))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    return {"git_commit": commit, "source_sha256": result.hexdigest()}


def safe_file(root, relative):
    root = root.resolve()
    path = (root / urllib.parse.unquote(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d+)-(\d+)", value or "")
    if not match:
        raise ValueError("one explicit inclusive byte range required")
    first, last = map(int, match.groups())
    if first > last or last >= size:
        raise ValueError("range outside immutable object")
    return first, last


def serve(package, manifest, config, host, port):
    manifest_file = safe_file(package, manifest)
    build = json.loads((OUT / "build_identity.json").read_text())
    wasm = OUT / "capture_lod_browser_bg.wasm"
    if digest(wasm) != build["wasm_sha256"]:
        raise ValueError("Wasm does not match frozen build identity")
    config = dict(config, manifest_url="/package/" + urllib.parse.quote(manifest),
                  manifest_sha256=digest(manifest_file), wasm_sha256=build["wasm_sha256"],
                  renderer_revision=build["renderer_revision"])
    config_bytes = json.dumps(config, allow_nan=False).encode()
    etags = {}
    etag_lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/qualification.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(config_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if body:
                    self.wfile.write(config_bytes)
                return
            try:
                if path.startswith("/package/"):
                    file = safe_file(package, path.removeprefix("/package/"))
                else:
                    file = safe_file(REPO / "www", path.lstrip("/") or "lod-qualification.html")
                stat = file.stat()
                key = (file, stat.st_size, stat.st_mtime_ns)
                with etag_lock:
                    etag = etags.get(key)
                    if etag is None:
                        etag = etags[key] = '"' + digest(file) + '"'
                if self.headers.get("If-Match") not in {None, etag}:
                    self.send_error(412, "immutable object version differs")
                    return
                first, last = 0, stat.st_size - 1
                partial = self.headers.get("Range") is not None
                if partial:
                    first, last = byte_range(self.headers.get("Range"), stat.st_size)
                self.send_response(206 if partial else 200)
                self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(last - first + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-store")
                if partial:
                    self.send_header("Content-Range", f"bytes {first}-{last}/{stat.st_size}")
                self.end_headers()
                if body:
                    with file.open("rb") as stream:
                        stream.seek(first)
                        remaining = last - first + 1
                        while remaining:
                            chunk = stream.read(min(65536, remaining))
                            if not chunk:
                                raise OSError("immutable object truncated during response")
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
            except FileNotFoundError:
                self.send_error(404)
            except ValueError:
                self.send_error(416)

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"Qualification page: http://{host}:{server.server_port}/lod-qualification.html", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-identity", action="store_true")
    parser.add_argument("--write-build-identity", type=pathlib.Path)
    parser.add_argument("--features")
    parser.add_argument("--package", type=pathlib.Path)
    parser.add_argument("--manifest", default="scene.gsplatlod")
    parser.add_argument("--config", type=pathlib.Path, default=REPO / "tools/fixtures/lod_browser_capture.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.source_identity:
        print(json.dumps(source_identity(), sort_keys=True))
    elif args.write_build_identity:
        before = json.loads(args.write_build_identity.read_text())
        if before != source_identity():
            raise SystemExit("Sources changed during the build; artifact identity cannot be attested")
        before.update(wasm_sha256=digest(OUT / "capture_lod_browser_bg.wasm"), features=args.features)
        before["renderer_revision"] = before["git_commit"] + "+source-sha256-" + before["source_sha256"]
        (OUT / "build_identity.json").write_text(json.dumps(before, indent=2) + "\n")
    else:
        if args.package is None:
            parser.error("--package DIRECTORY is required for serving")
        serve(args.package, args.manifest, json.loads(args.config.read_text()), args.host, args.port)


if __name__ == "__main__":
    main()
