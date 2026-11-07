"""Talk to Fusion's local MCP server (Preferences > General > API > Fusion MCP Server).

Runs a Fusion API script from a file and prints what it printed, or saves a screenshot of the
active viewport. Standard library only, so it runs with any python3.

    python3 hardware/fusion.py run hardware/fusion_scripts/version.py
    python3 hardware/fusion.py shot runs/cad/view.png
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:27182/mcp"
PARAMS = Path(__file__).with_name("tars_params.json")


class Fusion:
    def __init__(self, url: str = URL) -> None:
        self.url, self.session, self.next_id = url, None, 0
        self._post({"method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "tars-rl", "version": "1"}}})
        self._post({"method": "notifications/initialized"}, notify=True)

    def _post(self, msg: dict, notify: bool = False) -> dict:
        msg = {"jsonrpc": "2.0", **msg}
        if not notify:
            self.next_id += 1
            msg["id"] = self.next_id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.session = resp.headers.get("Mcp-Session-Id") or self.session
                body = resp.read().decode()
        except OSError as e:
            raise SystemExit(f"Fusion's MCP server is not answering at {self.url} ({e}); "
                             "is Fusion open with Preferences > General > API > Fusion MCP Server ticked?")
        if notify or not body.strip():
            return {}
        if body.lstrip().startswith("data:"):  # event-stream answer: take the last data line
            body = [ln[5:] for ln in body.splitlines() if ln.startswith("data:")][-1]
        out = json.loads(body)
        if "error" in out:
            raise SystemExit(f"Fusion MCP error: {out['error']}")
        return out["result"]

    def call(self, tool: str, args: dict) -> list[dict]:
        result = self._post({"method": "tools/call", "params": {"name": tool, "arguments": args}})
        if result.get("isError"):
            text = "\n".join(c.get("text", "") for c in result.get("content", []))
            raise SystemExit(f"Fusion script failed:\n{text}")
        return result.get("content", [])

    def run(self, script: str, read_only: bool = False) -> str:
        """Run a script defining run(_context); the parameter file's JSON is available as PARAMS_JSON."""
        source = f"PARAMS_JSON = {PARAMS.read_text()!r}\n" + script
        obj = {"script": source, **({"readOnly": True} if read_only else {})}
        content = self.call("fusion_mcp_execute", {"featureType": "script", "object": obj})
        return "\n".join(c.get("text", "") for c in content)

    def screenshot(self, out: Path) -> Path:
        content = self.call("fusion_mcp_read", {"queryType": "screenshot"})
        for c in content:
            if c.get("type") == "image":
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(c["data"]))
                return out
        raise SystemExit("Fusion returned no image: " + json.dumps(content)[:500])


def main(argv: list[str]) -> None:
    if len(argv) < 2 or argv[0] not in ("run", "shot"):
        raise SystemExit(__doc__)
    fusion = Fusion()
    if argv[0] == "run":
        print(fusion.run(Path(argv[1]).read_text(), read_only="--read-only" in argv))
    else:
        print(fusion.screenshot(Path(argv[1])))


if __name__ == "__main__":
    main(sys.argv[1:])
