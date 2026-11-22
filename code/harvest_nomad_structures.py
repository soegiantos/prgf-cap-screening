#!/usr/bin/env python3
"""Harvest NOMAD structures (cell + atomic positions) for formulas that
have no MP/COD structure — serverfisika job.

Why this script exists: the NOMAD *entry* document (entries/query) only
carries a summarised cell (a/b/c/angles) and NO atomic positions.
Positions live in the archive: run[0].system[0].atoms
(positions + lattice_vectors in SI metres, labels as element symbols).

Access path (found by probing 2026-11-18, after three dead ends):
- POST /entries/archive/query with the NESTED-dict `required` format
  {"run": {"system": {"atoms": "*"}}}
- The {"include": [...]} list format returns an EMPTY archive - do not use.
- OPTIMADE on this deployment 404s - do not use.
- Per-entry /entries/{id}/archive/query works too but is 1 request/entry;
  the bulk endpoint does 50 entries per request.
- Which system: topology[0] ("original", NOMAD's representative) points
  at #/run/0/system/0/atoms. Use the first system with valid atoms.
  (Most entries have exactly 1 run / 1 system.)

Scope: data_raw/nomad_struct_targets.json - 1,737 formulas with NOMAD
entries but no MP/COD structure (1,215 NOMAD-only + 522 NOMAD+AFLOW/
PubChem), up to 3 entry_ids each (3,397 entries). One structure per
formula: the first entry_id (in listed order) that yields a valid
structure wins.

Output: data_raw/nomad_structures.jsonl, one record per formula:
  formula (target, from consolidated v2), entry_id,
  formula_struct (reduced formula recomputed from the structure labels),
  formula_match (bool: structure composition == target composition),
  n_atoms, lattice_vectors (Angstrom, 3x3), positions (Angstrom,
  cartesian), labels. Metres x1e10 -> Angstrom.
Validation: labels/positions counts agree, finite numbers,
cell volume > 1 A^3, n_atoms <= 300 (screening cap).
Resume-aware: formulas already in the output file are skipped.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/archive/query"
BATCH = 50
MAX_ATOMS = 300
M_TO_A = 1e10

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(f):
    """'CaH4O6P' -> {'Ca': 1, 'H': 4, 'O': 6, 'P': 1} (reduced already)."""
    counts = {}
    for el, n in FORMULA_RE.findall(f or ""):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts


def reduce_counts(counts):
    g = 0
    for v in counts.values():
        g = math.gcd(g, v)
    if g > 1:
        counts = {k: v // g for k, v in counts.items()}
    return counts


def formula_str(counts):
    # canonical: Ca first, then alphabetical (matches consolidated style)
    els = sorted(counts, key=lambda e: (e != "Ca", e))
    return "".join(e + (str(counts[e]) if counts[e] != 1 else "") for e in els)


def cell_volume(lv):
    (a, b, c) = lv
    cx = b[1] * c[2] - b[2] * c[1]
    cy = b[2] * c[0] - b[0] * c[2]
    cz = b[0] * c[1] - b[1] * c[0]
    return abs(a[0] * cx + a[1] * cy + a[2] * cz)


def finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def extract_structure(entry):
    """Return (record_dict, reason) - record_dict or None."""
    try:
        runs = entry.get("archive", {}).get("run", [])
    except Exception:
        return None, "no_archive"
    if not runs:
        return None, "no_run"
    for run in runs:
        for sys in run.get("system", []) or []:
            atoms = (sys or {}).get("atoms") or {}
            labels = atoms.get("labels")
            pos = atoms.get("positions")
            lv = atoms.get("lattice_vectors")
            if not labels or not pos or not lv:
                continue
            if not (len(labels) == len(pos)):
                continue
            if len(labels) < 1 or len(labels) > MAX_ATOMS:
                return None, "over_cap" if len(labels) > MAX_ATOMS else "no_atoms"
            try:
                lv_a = [[float(x) * M_TO_A for x in row] for row in lv]
                pos_a = [[float(x) * M_TO_A for x in row] for row in pos]
            except Exception:
                continue
            if len(lv_a) != 3 or any(len(r) != 3 for r in lv_a):
                continue
            if any(len(r) != 3 for r in pos_a):
                continue
            if not all(finite(x) for row in lv_a + pos_a for x in row):
                continue
            if cell_volume(lv_a) <= 1.0:
                continue
            counts = reduce_counts(dict(Counter(labels)))
            return {
                "n_atoms": len(labels),
                "lattice_vectors": lv_a,
                "positions": pos_a,
                "labels": list(labels),
                "formula_struct": formula_str(counts),
                "_counts": counts,
            }, None
    return None, "no_atoms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(BASE_DIR, "data_raw", "nomad_struct_targets.json"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data_raw", "nomad_structures.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only first N formulas (test)")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    targets = json.load(open(args.targets))
    if args.limit:
        targets = targets[: args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["formula"])
                except Exception:
                    pass
    pending = [t for t in targets if t["formula"] not in done]
    print(f"target: {len(targets)} rumus, sudah selesai: {len(done)}, "
          f"menunggu: {len(pending)}", flush=True)
    if not pending:
        print("SELESAI: tidak ada yang menunggu", flush=True)
        return

    eid_to_formula = {}
    all_ids = []
    for t in pending:
        for eid in t["entry_ids"]:
            if eid not in eid_to_formula:
                eid_to_formula[eid] = t["formula"]
                all_ids.append(eid)
    print(f"entry_ids yang akan diambil: {len(all_ids)}", flush=True)

    stats = Counter()
    failed = []
    out_f = open(args.out, "a")
    try:
        for bi in range(0, len(all_ids), args.batch):
            batch = all_ids[bi:bi + args.batch]
            d = None
            for attempt in range(3):
                try:
                    rr = requests.post(URL, json={
                        "query": {"entry_id": {"any": batch}},
                        "required": {"run": {"system": {"atoms": "*"}}},
                        "pagination": {"page_size": len(batch)},
                    }, timeout=120, headers=UA)
                    if rr.status_code == 200:
                        d = rr.json()
                        break
                    print(f"  batch {bi//args.batch+1} HTTP {rr.status_code}: {rr.text[:160]}", flush=True)
                except Exception as e:
                    print(f"  batch {bi//args.batch+1} err {e}", flush=True)
                time.sleep(2 * (attempt + 1))
            if d is None:
                print(f"  batch {bi//args.batch+1} GAGAL setelah 3 percobaan", flush=True)
                stats["batch_gagal"] += 1
                continue
            got = {e.get("entry_id"): e for e in d.get("data", [])}
            for eid in batch:
                formula = eid_to_formula.get(eid)
                if formula in done:
                    continue
                entry = got.get(eid)
                if entry is None:
                    stats["entry_hilang"] += 1
                    continue
                rec, reason = extract_structure(entry)
                if rec is None:
                    stats["tanpa_struktur_" + (reason or "?")] += 1
                    continue
                counts = rec.pop("_counts")
                target_counts = parse_formula(formula)
                match = (counts == target_counts)
                out_f.write(json.dumps({
                    "formula": formula,
                    "entry_id": eid,
                    "formula_struct": rec["formula_struct"],
                    "formula_match": match,
                    "n_atoms": rec["n_atoms"],
                    "lattice_vectors": rec["lattice_vectors"],
                    "positions": rec["positions"],
                    "labels": rec["labels"],
                }) + "\n")
                out_f.flush()
                done.add(formula)
                stats["struktur_ok"] += 1
                if not match:
                    stats["rumus_beda"] += 1
            if (bi // args.batch + 1) % 5 == 0 or bi + args.batch >= len(all_ids):
                print(f"  batch {bi//args.batch+1}/{math.ceil(len(all_ids)/args.batch)}: "
                      f"selesai {len(done)}/{len(targets)} rumus; {dict(stats)}", flush=True)
            time.sleep(0.3)
    finally:
        out_f.close()

    still = [t["formula"] for t in targets if t["formula"] not in done]
    summary = {
        "target_rumus": len(targets),
        "selesai": len(done),
        "gagal_rumus": len(still),
        "gagal_rumus_contoh": still[:20],
        "stats": dict(stats),
    }
    with open(os.path.join(os.path.dirname(args.out), "nomad_struct_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("SELESAI: " + json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
