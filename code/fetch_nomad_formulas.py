#!/usr/bin/env python3
"""Enrich the 8000 NOMAD continuation entries with formula fields.

Problem: harvest_nomad_cont.py saved only entry_id / formula / elements,
and NOMAD returns `results.material.formula` as null for these entries.
consolidate.py needs `formula_reduced` (the field the original 2000
entries carry), so without this fix all 8000 new entries would be
silently dropped as unparseable.

Fix: re-query NOMAD by entry_id in batches of 100, requesting the
material fields, and rewrite nomad_ca_p_cont.jsonl in the SAME shape as
the original nomad_ca_p.jsonl: entry_id, formula_hill, formula_reduced,
elements. The thin file is backed up first as nomad_ca_p_cont_thin.jsonl.
Entries that still have no formula after the re-query are kept with
nulls and counted (they will be reported, not silently dropped).
"""
import json, os, time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw")
UA = {"User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
CONT_PATH = os.path.join(OUT_DIR, "nomad_ca_p_cont.jsonl")
THIN_PATH = os.path.join(OUT_DIR, "nomad_ca_p_cont_thin.jsonl")
BATCH = 50


def main():
    src = CONT_PATH if os.path.exists(CONT_PATH) else THIN_PATH
    ids = []
    for line in open(src):
        try:
            ids.append(json.loads(line).get("entry_id"))
        except Exception:
            pass
    ids = [i for i in ids if i]
    print(f"entry_id lanjutan (dari {os.path.basename(src)}): {len(ids)}", flush=True)
    # backup thin file
    if not os.path.exists(THIN_PATH):
        os.replace(CONT_PATH, THIN_PATH)
        print("thin file dibackup ke nomad_ca_p_cont_thin.jsonl", flush=True)
    else:
        print("backup thin sudah ada", flush=True)

    enriched, no_formula = [], 0
    for bi in range(0, len(ids), BATCH):
        batch = ids[bi:bi + BATCH]
        # TANPA `required`: path formula bukan "doc quantity" NOMAD
        # (HTTP 422), jadi ambil entri penuh lalu petik field-nya.
        # Nama field di entri penuh: chemical_formula_hill/reduced.
        payload = {"query": {"entry_id": {"any": batch}},
                   "pagination": {"page_size": BATCH}}
        ok = False
        for attempt in range(3):
            try:
                rr = requests.post(URL, json=payload, timeout=120, headers=UA)
                if rr.status_code == 200:
                    ok = True
                    break
                print(f"  batch {bi//BATCH+1} HTTP {rr.status_code}: {rr.text[:160]}", flush=True)
            except Exception as e:
                print(f"  batch {bi//BATCH+1} err {e}", flush=True)
            time.sleep(2 * (attempt + 1))
        if not ok:
            print(f"  batch {bi//BATCH+1} GAGAL setelah 3 percobaan — entri dilewati", flush=True)
            continue
        d = rr.json()
        got = {e.get("entry_id"): e for e in d.get("data", [])}
        for eid in batch:
            e = got.get(eid)
            mat = (e.get("results") or {}).get("material", {}) if e else {}
            fr = mat.get("chemical_formula_reduced") or mat.get("formula_reduced")
            fh = mat.get("chemical_formula_hill") or mat.get("formula_hill")
            els = mat.get("elements")
            if not fr and not fh:
                no_formula += 1
            enriched.append({"entry_id": eid, "formula_hill": fh,
                             "formula_reduced": fr, "elements": els})
        if (bi // BATCH + 1) % 10 == 0:
            print(f"  batch {bi//BATCH+1}: total {len(enriched)}, tanpa rumus {no_formula}", flush=True)
        time.sleep(0.3)

    with open(CONT_PATH, "w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")
    print(f"SELESAI: {len(enriched)} entri ditulis ke nomad_ca_p_cont.jsonl; "
          f"tanpa rumus: {no_formula}", flush=True)


if __name__ == "__main__":
    main()
