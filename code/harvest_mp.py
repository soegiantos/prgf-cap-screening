#!/usr/bin/env python3
"""Harvest Materials Project for PRGF baru — all materials containing Ca and P.
Key is read from ~/.secrets/mp_api_key (never printed, never written to output).
API: https://api.materialsproject.org/materials/summary/  (CC-BY 4.0)
Total for elements=Ca,P checked 2026-11-17: 815 documents.
"""
import json, os, time
from datetime import datetime, timezone
import requests

KEY_PATH = os.path.expanduser("~/.secrets/mp_api_key")
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data_raw"
os.makedirs(OUT_DIR, exist_ok=True)

key = open(KEY_PATH).read().strip()
headers = {"X-API-KEY": key, "User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
FIELDS = ("material_id,formula_pretty,formula_anonymous,chemsys,nsites,volume,density,"
          "density_atomic,symmetry,formation_energy_per_atom,energy_above_hull,is_stable,"
          "band_gap,efermi,total_magnetization,theoretical,elements")
BASE = "https://api.materialsproject.org/materials/summary/"

rows, seen = [], set()
skip = 0
per_page = 500
while True:
    ok = False
    for attempt in range(3):
        try:
            r = requests.get(BASE, params={"elements": "Ca,P", "_limit": per_page,
                                           "_skip": skip, "_fields": FIELDS},
                             headers=headers, timeout=60)
            if r.status_code == 200:
                d = r.json(); ok = True; break
            print(f"MP skip={skip} attempt {attempt+1} HTTP {r.status_code}: {r.text[:160]}", flush=True)
        except Exception as e:
            print(f"MP skip={skip} attempt {attempt+1} err {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    if not ok:
        print("MP FAILED at skip", skip, flush=True)
        break
    data = d.get("data", [])
    total_doc = d.get("meta", {}).get("total_doc")
    print(f"MP skip={skip}: {len(data)} entries, total_doc={total_doc}", flush=True)
    for e in data:
        if e.get("material_id") in seen:
            print("DUPLICATE", e.get("material_id"), flush=True)
        seen.add(e.get("material_id"))
        rows.append(e)
    skip += len(data)
    if len(data) < per_page or (total_doc and skip >= total_doc):
        break
    time.sleep(0.5)

with open(os.path.join(OUT_DIR, "mp_ca_p.jsonl"), "w") as f:
    for e in rows:
        f.write(json.dumps(e) + "\n")
with open(os.path.join(OUT_DIR, "mp_ca_p.json"), "w") as f:
    json.dump(rows, f)

print("MP TOTAL", len(rows), "UNIQUE", len(seen))

audit_path = os.path.join(OUT_DIR, "audit.json")
audit = json.load(open(audit_path)) if os.path.exists(audit_path) else {"sources": {}}
audit["sources"]["Materials Project"] = {
    "query": "elements=Ca,P (materials containing Ca and P)",
    "entries": len(rows),
    "unique_material_ids": len(seen),
    "files": ["mp_ca_p.jsonl", "mp_ca_p.json"],
    "license": "CC-BY 4.0",
    "url": BASE,
    "note": "Key read from ~/.secrets/mp_api_key (owner-supplied 2026-11-17); key value never stored in output files, logs, or this audit. Total 815 vs 57 in the old pipeline, which queried only narrow Ca-P systems.",
}
audit["total_records_harvested"] = audit.get("total_records_harvested", 0) + len(rows)
audit["mp_harvested_utc"] = datetime.now(timezone.utc).isoformat()
audit["not_included"].pop("Materials Project", None)
json.dump(audit, open(audit_path, "w"), indent=2)
print("Audit updated. New total_records_harvested:", audit["total_records_harvested"])
