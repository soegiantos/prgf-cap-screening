#!/usr/bin/env python3
"""Ambil struktur kristal NOMAD untuk 10.000 entri Ca+P yang sudah dipanen.

Latar: screening GNN putaran 2 (18 Nov 2026, Mac) hanya menyaring 1.735
struktur (815 MP + 920 COD) — cakupan rumus unik 42,1% dari 3.249.
NOMAD (10.000 entri) + AFLOW (5.874) di berkas harvest TIDAK menyimpan
struktur, jadi 57,9% rumus tidak bisa disaring GNN. Skrip ini menutup
celah NOMAD: ambil `results.material.structure` per entri, supaya
screening putaran 3 bisa memakai GNN pretrain di seluruh pool NOMAD.

Pola mengikuti fetch_nomad_formulas.py (terbukti 8.000/8.000 berhasil):
re-query per entry_id batch 50, TANPA `required` (path material bukan
"doc quantity" NOMAD — HTTP 422 bila diminta), ambil entri penuh lalu
petik field-nya. Nama field di entri penuh: results.material.structure
(dict: lattice_vectors + atoms). Retry 3x per batch; bisa dilanjutkan
(resume): entri yang sudah ada di berkas output dilewati.

Jalankan di Mac Mini (paralel dengan DFT r1b — kerja ini dominan
jaringan, bukan CPU):  nohup python3 fetch_nomad_structures.py > fetch_nomad_structures.log 2>&1 &
Perkiraan: 10.000 entri / 50 = 200 batch, ~0,3 dtk jeda + waktu respons
API — hitungan jam, bukan hari. Output: data_raw/nomad_ca_p_structures.jsonl
satu baris per entri: {entry_id, formula_reduced, structure}.
Entri tanpa struktur tetap ditulis dengan structure=null + dihitung
(dilaporkan, tidak dibuang diam-diam).
"""
import json, os, time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data_raw")
UA = {"User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
URL = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
OUT_PATH = os.path.join(OUT_DIR, "nomad_ca_p_structures.jsonl")
BATCH = 50


def load_ids():
    ids = []
    seen_order = []
    for path in (os.path.join(OUT_DIR, "nomad_ca_p.jsonl"),
                 os.path.join(OUT_DIR, "nomad_ca_p_cont.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        eid = json.loads(line).get("entry_id")
                    except Exception:
                        continue
                    if eid and eid not in seen_order:
                        seen_order.append(eid)
        except FileNotFoundError:
            print(f"  (tidak ada {os.path.basename(path)})", flush=True)
    return seen_order


def load_done():
    done = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r.get("entry_id")] = r
                except Exception:
                    pass
    return done


def main():
    ids = load_ids()
    print(f"entry_id NOMAD total: {len(ids)}", flush=True)
    done = load_done()
    print(f"sudah ada di {os.path.basename(OUT_PATH)}: {len(done)}", flush=True)
    todo = [i for i in ids if i not in done]
    print(f"masih harus diambil: {len(todo)}", flush=True)
    if not todo:
        print("SELESAI: semua entri sudah punya struktur.", flush=True)
        return

    no_struct = 0
    written = 0
    with open(OUT_PATH, "a") as f:
        for bi in range(0, len(todo), BATCH):
            batch = todo[bi:bi + BATCH]
            payload = {"query": {"entry_id": {"any": batch}},
                       "pagination": {"page_size": BATCH}}
            ok = False
            for attempt in range(3):
                try:
                    rr = requests.post(URL, json=payload, timeout=120, headers=UA)
                    if rr.status_code == 200:
                        ok = True
                        break
                    print(f"  batch {bi//BATCH+1} HTTP {rr.status_code}: {rr.text[:160]}",
                          flush=True)
                except Exception as e:
                    print(f"  batch {bi//BATCH+1} err {e}", flush=True)
                time.sleep(2 * (attempt + 1))
            if not ok:
                print(f"  batch {bi//BATCH+1} GAGAL setelah 3 percobaan — "
                      f"entri dilewati (bisa resume)", flush=True)
                continue
            d = rr.json()
            got = {e.get("entry_id"): e for e in d.get("data", [])}
            for eid in batch:
                e = got.get(eid)
                mat = (e.get("results") or {}).get("material", {}) if e else {}
                struct = mat.get("structure")
                fr = mat.get("chemical_formula_reduced") or mat.get("formula_reduced")
                if not struct:
                    no_struct += 1
                rec = {"entry_id": eid, "formula_reduced": fr,
                       "structure": struct}
                f.write(json.dumps(rec) + "\n")
                written += 1
            f.flush()
            if (bi // BATCH + 1) % 10 == 0:
                print(f"  batch {bi//BATCH+1}: total ditulis {written}, "
                      f"tanpa struktur {no_struct}", flush=True)
            time.sleep(0.3)

    print(f"SELESAI: +{written} entri ditulis ke nomad_ca_p_structures.jsonl; "
          f"tanpa struktur: {no_struct}", flush=True)


if __name__ == "__main__":
    main()
