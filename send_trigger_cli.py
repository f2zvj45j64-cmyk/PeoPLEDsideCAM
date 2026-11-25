#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI trigger sender (new JSON format) + timing measurement"""

import argparse
import time
import requests
import json
import sys

def default_sn():
    return "SN" + time.strftime("%Y%m%d%H%M%S")

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="http://127.0.0.1:9000")
ap.add_argument("--pos1", default="1")
ap.add_argument("--pos2", default="2")
ap.add_argument("--pos3", default="DM11_XXX")
ap.add_argument("--sn1", default=None)
ap.add_argument("--sn2", default=None)
ap.add_argument("--sn3", default="NA")
args = ap.parse_args()

sn1 = args.sn1 or default_sn()
sn2 = args.sn2 or default_sn()
sn3 = args.sn3 or "NA"

payload = {
    "data1": {"position": str(args.pos1), "SN": sn1, "result": "NA", "failure symptoms": "NA"},
    "data2": {"position": str(args.pos2), "SN": sn2, "result": "NA", "failure symptoms": "NA"},
    "data3": {"position": str(args.pos3), "SN": sn3, "result": "NA", "failure symptoms": "NA"}
}

url = args.host.rstrip("/") + "/api/eval"

print("Sending to:", url)
print("Payload:", json.dumps(payload, indent=2))

try:
    t0 = time.perf_counter()

    r = requests.post(url, json=payload, timeout=30)

    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    print("\n=== RESPONSE ===")
    print("HTTP:", r.status_code)
    print("Response size:", len(r.content), "bytes")
    print("Round-trip time: %.2f ms" % elapsed_ms)

    try:
        print("JSON:", json.dumps(r.json(), indent=2))
    except Exception:
        print("Raw text:", r.text)

except Exception as e:
    print("ERROR:", e)
    sys.exit(1)
