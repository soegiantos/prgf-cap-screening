#!/usr/bin/env python3
"""Harvest Materials Project — ALL phosphates (P + O, any cations) for GNN pretraining.

Gate G1 (WP2) needs more labelled structure data before the GNN can be used
for DFT selection. This fetches a broad phosphate pretraining set to
transfer-learn from, then fine-tune on the 815 Ca-P structures.

Key is read from ~/.secrets/mp_api_key (never printed, never written to output).
API: https://api.materialsproject.org/materials/summary/ (CC-BY 4.0)

Phase 1: summaries for elements=P,O (paginated).
Phase 2: structures for a subset (formation energy present, nsites <= 120,
         NOT in the existing 815 Ca-P set), cap 5000, batches of 100 ids.
         Output shape matches data_raw/mp_structures.json so the WP2
         training scripts can reuse it directly.
"""
import json, os, sys, time
import requests

KEY_PATH = os.path.expanduser("~/.secrets/mp_api_key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw")
os.makedirs(OUT_DIR, exist_ok=True)

key = open(KEY_PATH).read().strip()
headers = {"X-API-KEY": key,
           "User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
BASE = "https://api.materialsproject.org/materials/summary/"
SUM_FIELDS = ("material_id,formula_pretty,formula_anonymous,chemsys,nsites,volume,density,"
              "density_atomic,symmetry,formation_energy_per_atom,energy_above_hull,is_stable,"
              "band_gap,efermi,total_magnetization,theoretical,elements")
STRUCT_FIELDS = ("material_id,formula_pretty,nsites,density,formation_energy_per_atom,"
                 "energy_above_hull,band_gap,structure")
MAX_STRUCTURES = 5000
MAX_NSITES = 120
BATCH = 100


def get(params, tries=3):
    for attempt in range(tries):
        try:
            r = requests.get(BASE, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  MP HTTP {r.status_code}: {r.text[:160]}", flush=True)
        except Exception as e:
            print(f"  MP err {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    return None


def phase1():
    out_path = os.path.join(OUT_DIR, "mp_phosphate_summaries.jsonl")
    rows, seen, skip, per_page = [], set(), 0, 500
    while True:
        d = get({"elements": "P,O", "_limit": per_page, "_skip": skip, "_fields": SUM_FIELDS})
        if d is None:
            print("MP phase1 FAILED at skip", skip, flush=True)
            break
        data = d.get("data", [])
        total_doc = d.get("meta", {}).get("total_doc")
        print(f"MP summaries skip={skip}: {len(data)} entries, total_doc={total_doc}", flush=True)
        for e in data:
            if e.get("material_id") in seen:
                print("DUPLICATE", e.get("material_id"), flush=True)
            seen.add(e.get("material_id"))
            rows.append(e)
        skip += len(data)
        if len(data) < per_page or (total_doc and skip >= total_doc):
            break
        time.sleep(0.3)
    with open(out_path, "w") as f:
        for e in rows:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(OUT_DIR, "mp_phosphate_summaries.json"), "w") as f:
        json.dump(rows, f)
    print("MP PHASE1 TOTAL", len(rows), "UNIQUE", len(seen), flush=True)
    return rows


def phase2(rows):
    # existing Ca-P training ids (exclude so pretraining is clean transfer)
    existing = set()
    try:
        for d in json.load(open(os.path.join(OUT_DIR, "mp_structures.json"))):
            existing.add(d.get("material_id"))
    except Exception as e:
        print("  (could not read existing mp_structures.json:", e, ")", flush=True)
    print(f"  existing Ca-P ids to exclude: {len(existing)}", flush=True)

    qual = [e for e in rows
            if e.get("formation_energy_per_atom") is not None
            and (e.get("nsites") or 999) <= MAX_NSITES
            and e.get("material_id") not in existing]
    print(f"  qualifying (E_form present, nsites<={MAX_NSITES}, not Ca-P): {len(qual)}", flush=True)
    # deterministic strided sample if over cap (preserves size diversity)
    qual.sort(key=lambda e: (e.get("nsites") or 0, e.get("material_id") or ""))
    if len(qual) > MAX_STRUCTURES:
        step = len(qual) / MAX_STRUCTURES
        qual = [qual[int(i * step)] for i in range(MAX_STRUCTURES)]
    print(f"  selected for structure fetch: {len(qual)}", flush=True)

    out_path = os.path.join(OUT_DIR, "mp_phosphate_structures.json")
    done = []
    done_ids = set()
    if os.path.exists(out_path):
        try:
            done = json.load(open(out_path))
            done_ids = {d.get("material_id") for d in done}
            print(f"  resuming: {len(done)} structures already fetched", flush=True)
        except Exception:
            done, done_ids = [], set()
    todo = [e for e in qual if e.get("material_id") not in done_ids]
    print(f"  to fetch: {len(todo)}", flush=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        ids = ",".join(e["material_id"] for e in batch)
        d = get({"material_ids": ids, "_limit": BATCH, "_fields": STRUCT_FIELDS})
        if d is None:
            print(f"  batch {i//BATCH+1} FAILED, stopping (progress saved)", flush=True)
            break
        data = d.get("data", [])
        for e in data:
            if e.get("structure"):
                done.append(e)
                done_ids.add(e.get("material_id"))
        print(f"  batch {i//BATCH+1}: +{len(data)} structures, total {len(done)}", flush=True)
        with open(out_path, "w") as f:
            json.dump(done, f)
        time.sleep(0.3)
    print("MP PHASE2 TOTAL STRUCTURES", len(done), flush=True)


if __name__ == "__main__":
    rows = phase1()
    if rows:
        phase2(rows)
