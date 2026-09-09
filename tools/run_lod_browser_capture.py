#!/usr/bin/env python3
"""Opt-in actual browser/GPU run via local ChromeDriver, using no Python packages."""
import argparse
import base64
import json
import pathlib
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from check_lod_browser_capture import validate


def request(base, method, path, value=None):
    data = None if value is None else json.dumps(value, allow_nan=False).encode()
    call = urllib.request.Request(base + path, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=30) as response:
        body = response.read(32 * 1024 * 1024 + 1)
        if len(body) > 32 * 1024 * 1024:
            raise ValueError("WebDriver response ceiling exceeded")
        result = json.loads(body)["value"]
        if isinstance(result, dict) and "error" in result and "message" in result:
            raise ValueError(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/lod-qualification.html")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--chromedriver", default=shutil.which("chromedriver"))
    parser.add_argument("--chrome")
    parser.add_argument("--driver-port", type=int, default=9515)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--allow-software", action="store_true", help="Accept distinctly labeled software diagnostics; never hardware-qualified")
    parser.add_argument("--chrome-arg", action="append", default=[], help="Additional explicit browser flag; recorded with the run")
    args = parser.parse_args()
    if not args.chromedriver:
        parser.error("ChromeDriver is required; pass --chromedriver PATH")
    if not 1 <= args.timeout_seconds <= 1800:
        parser.error("timeout must be 1..1800 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.driver_port}"
    session = None
    with (args.output / "chromedriver.log").open("w") as log:
        driver = subprocess.Popen([args.chromedriver, f"--port={args.driver_port}"], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    request(base, "GET", "/status")
                    break
                except (urllib.error.URLError, ConnectionError):
                    if driver.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("ChromeDriver did not become ready")
                    time.sleep(0.1)
            flags = ["--enable-unsafe-webgpu", "--force-device-scale-factor=1", "--disable-dev-shm-usage"]
            if not args.headed:
                flags.append("--headless=new")
            flags.extend(args.chrome_arg)
            (args.output / "launch.json").write_text(json.dumps({"flags": flags, "url": args.url, "timeout_seconds": args.timeout_seconds, "allow_software": args.allow_software}, indent=2) + "\n")
            options = {"args": flags}
            if args.chrome:
                options["binary"] = args.chrome
            opened = request(base, "POST", "/session", {"capabilities": {
                "alwaysMatch": {"browserName": "chrome", "goog:chromeOptions": options,
                                "goog:loggingPrefs": {"browser": "ALL"}}}})
            session = opened["sessionId"]
            prefix = "/session/" + session
            (args.output / "browser_identity.json").write_text(json.dumps(opened["capabilities"], indent=2) + "\n")
            request(base, "POST", prefix + "/url", {"url": args.url})
            deadline = time.monotonic() + args.timeout_seconds
            status = None
            while time.monotonic() < deadline:
                status = request(base, "POST", prefix + "/execute/sync", {
                    "script": "const s=globalThis.__bgsBrowserQualification; return s ? {done:s.done,error:s.error} : null;", "args": []})
                if status and status["done"]:
                    break
                time.sleep(0.25)
            records = request(base, "POST", prefix + "/execute/sync", {
                "script": "return globalThis.__bgsBrowserQualification?.records ?? [];", "args": []})
            with (args.output / "evidence.jsonl").open("w") as capture:
                for record in records:
                    capture.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            screenshot = request(base, "GET", prefix + "/screenshot")
            (args.output / "browser.png").write_bytes(base64.b64decode(screenshot, validate=True))
            browser_logs = request(base, "POST", prefix + "/log", {"type": "browser"})
            (args.output / "browser_console.json").write_text(json.dumps(browser_logs, indent=2) + "\n")
            (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            if not status or not status["done"] or status["error"]:
                raise RuntimeError(f"browser incomplete: {status}")
            result = validate(records, allow_software=args.allow_software)
            (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            if session:
                try:
                    request(base, "DELETE", "/session/" + session)
                except (OSError, ValueError):
                    pass
            driver.terminate()
            try:
                driver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                driver.kill()
                driver.wait()


if __name__ == "__main__":
    main()
