#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI trigger sender (new JSON format)

Example:
  python3 send_trigger_cli.py \
    --host http://127.0.0.1:9000 \
    --pos1 1 --sn1 DM1_XXX \
    --pos2 DM6_XXX --sn2 NA \
    --pos3 DM11_XXX --sn3 NA
"""
import argparse, time, requests, json

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
  "data1":{"position":str(args.pos1),"SN":sn1,"result":"NA","failure symptoms":"NA"},
  "data2":{"position":str(args.pos2),"SN":sn2,"result":"NA","failure symptoms":"NA"},
  "data3":{"position":str(args.pos3),"SN":sn3,"result":"NA","failure symptoms":"NA"}
}
url = args.host.rstrip("/") + "/api/eval"
r = requests.post(url, json=payload, timeout=20)
print("HTTP", r.status_code)
print(r.text)
