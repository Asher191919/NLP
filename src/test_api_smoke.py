"""Smoke-test an OpenAI-compatible endpoint (default: dots studio).

Usage (PowerShell):
  $env:ONTOBRIDG_BASE_URL='https://note3-prev-api.askdiandian.com/v1'
  $env:ONTOBRIDG_MODEL='dots3-note-prev'
  $env:ONTOBRIDG_API_KEY='ak_...'   # or OPENAI_API_KEY
  & $env:MIMO_PYTHON D:\\NLP\\src\\test_api_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_json_cfg() -> dict:
    path = SRC / "dots_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_CFG = _load_json_cfg()
BASE = (
    os.environ.get("ONTOBRIDG_BASE_URL")
    or _CFG.get("base_url")
    or "https://note3-prev-api.askdiandian.com/v1"
).rstrip("/")
MODEL = os.environ.get("ONTOBRIDG_MODEL") or _CFG.get("model") or "dots3-note-prev"
KEY = (
    os.environ.get("ONTOBRIDG_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or (_CFG.get("api_key") if _CFG.get("api_key") else "")
    or ""
)


def get_models() -> None:
    url = f"{BASE}/models"
    req = urllib.request.Request(url, method="GET")
    if KEY:
        req.add_header("Authorization", f"Bearer {KEY}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print("GET /models ->", json.dumps(data, ensure_ascii=False)[:500])


def chat_once() -> dict:
    url = f"{BASE}/chat/completions"
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "Respond with compact JSON only."},
            {
                "role": "user",
                "content": (
                    "Return JSON only: {\"ok\": true, "
                    "\"echo\": \"ontology learning\"}"
                ),
            },
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = data["choices"][0]["message"]["content"]
    print("POST /chat/completions ok")
    print("model:", data.get("model") or MODEL)
    print("content:", text[:400])
    print("usage:", data.get("usage"))
    return data


def main() -> int:
    print(f"BASE={BASE}")
    print(f"MODEL={MODEL}")
    print(f"KEY set={bool(KEY)} (len={len(KEY)})")
    if not KEY:
        print("ERROR: 只差 API Key —— 请编辑下列文件中唯一空字段后重跑:")
        print("  $env:ONTOBRIDG_API_KEY = 'ak_<rotated-key>'")
        print("  或设置兼容变量 $env:OPENAI_API_KEY")
        print("然后:  & D:\\NLP\\src\\run_dots_smoke.ps1")
        return 2
    try:
        get_models()
    except Exception as exc:
        print(f"models probe failed (non-fatal): {exc}")
    try:
        chat_once()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        print(f"HTTP {exc.code}: {detail}")
        return 1
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
