#!/usr/bin/env python3
"""Fetch MP entries for ALL subsystems of the 13 WP3 candidates' chemistries.

Why: analisis_dft_r1.py builds a provisional hull from only 34 entries
because the MP `chemsys` query returns EXACT chemical systems only — a
query for Ca-P-O-Zn does not return the CaO, ZnO, Ca-P or P-O binaries
that actually bound the hull. To get a complete hull for each candidate
we need every non-empty subset of its element set (singles, binaries,
ternaries, ... up to the full system).

Reads the 13 selected candidates from wp3_round1_candidates.json
(`kandidat_terpilih`, each with a `counts` dict). Computes the union of
all subsystems, queries the MP summary endpoint per subsystem
(chemsys=A-B-C, hyphen-separated, alphabetical), and saves the
deduplicated entries to data_raw/mp_hull_subsystems.json with
formation_energy_per_atom (the quantity a PhaseDiagram needs).

Key is read from ~/.secrets/mp_api_key (never printed, never written).
API: https://api.materialsproject.org/materials/summary/ (CC-BY 4.0)
"""
import json, os, time
from itertools import combinations
import requests

KEY_PATH = os.path.expanduser("~/.secrets/mp_api_key")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw")
os.makedirs(OUT_DIR, exist_ok=True)

key = open(KEY_PATH).read().strip()
headers = {"X-API-KEY": key,
           "User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
BASE = "https://api.materialsproject.org/materials/summary/"
FIELDS = ("material_id,formula_pretty,chemsys,nsites,density,"
          "formation_energy_per_atom,energy_above_hull,is_stable,elements")


def get(params, tries=3):
    for attempt in range(tries):
        try:
            r = requests.get(BASE, params=params, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code}: {r.text[:160]}", flush=True)
        except Exception as e:
            print(f"  err {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    return None


def main():
    cands = json.load(open(os.path.join(BASE_DIR, "wp3_round1_candidates.json")))
    selected = cands["kandidat_terpilih"]
    subsystems = set()
    for c in selected:
        els = sorted(c["counts"].keys())
        for r in range(1, len(els) + 1):
            for sub in combinations(els, r):
                subsystems.add(sub)
    subsystems = sorted(subsystems, key=lambda s: (len(s), s))
    print(f"Kandidat: {len(selected)}; subsistem unik: {len(subsystems)}", flush=True)

    out_path = os.path.join(OUT_DIR, "mp_hull_subsystems.json")
    seen, rows = set(), []
    if os.path.exists(out_path):
        try:
            rows = json.load(open(out_path))
            seen = {d.get("material_id") for d in rows}
            print(f"  melanjutkan: {len(rows)} entri sudah ada", flush=True)
        except Exception:
            rows, seen = [], set()
    # subsystem mana yang sudah tercakup oleh entri yang ada
    done_sub = set()
    for d in rows:
        done_sub.add(tuple(sorted(d.get("elements", []))))
    todo = [s for s in subsystems if s not in done_sub]
    print(f"  subsistem yang masih perlu diambil: {len(todo)}", flush=True)

    for i, sub in enumerate(todo):
        chemsys = "-".join(sub)
        skip, per_page = 0, 1000
        n_sub = 0
        while True:
            d = get({"chemsys": chemsys, "_limit": per_page, "_skip": skip, "_fields": FIELDS})
            if d is None:
                print(f"  {chemsys}: GAGAL, lanjut ke subsistem berikut", flush=True)
                break
            data = d.get("data", [])
            total = d.get("meta", {}).get("total_doc")
            for e in data:
                if e.get("material_id") in seen:
                    continue
                seen.add(e.get("material_id"))
                rows.append(e)
                n_sub += 1
            skip += len(data)
            if len(data) < per_page or (total and skip >= total):
                break
            time.sleep(0.3)
        if (i + 1) % 20 == 0 or n_sub:
            print(f"  [{i+1}/{len(todo)}] {chemsys}: +{n_sub} (total {len(rows)})", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f)
        time.sleep(0.2)

    with open(out_path, "w") as f:
        json.dump(rows, f)
    with_e = sum(1 for e in rows if e.get("formation_energy_per_atom") is not None)
    print(f"SELESAI: {len(rows)} entri hull dari {len(subsystems)} subsistem; "
          f"{with_e} punya energi formasi", flush=True)


if __name__ == "__main__":
    main()
