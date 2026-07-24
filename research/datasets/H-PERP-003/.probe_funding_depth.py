"""One-shot probe: how far back does OKX public funding-rate-history go?"""
import datetime
import time
import requests

BASE = "https://www.okx.com/api/v5"
INST = "BTC-USDT-SWAP"

rows = []
after = None
pages = 0
while pages < 100:
    params = {"instId": INST, "limit": "100"}
    if after:
        params["after"] = after
    r = requests.get(BASE + "/public/funding-rate-history", params=params, timeout=30).json()
    chunk = r.get("data") or []
    if not chunk:
        print(f"STOP at page {pages}: empty")
        break
    rows.extend(chunk)
    oldest_in_chunk = min(int(x["fundingTime"]) for x in chunk)
    after = str(oldest_in_chunk)
    pages += 1
    time.sleep(0.15)

print(f"Funding pages: {pages}, total rows: {len(rows)}")
if rows:
    o = min(int(x["fundingTime"]) for x in rows)
    n = max(int(x["fundingTime"]) for x in rows)
    print(f"span: {datetime.datetime.utcfromtimestamp(o/1000).isoformat()}")
    print(f"   to {datetime.datetime.utcfromtimestamp(n/1000).isoformat()}")
    print(f"days: {(n - o) / (1000 * 86400):.1f}")
