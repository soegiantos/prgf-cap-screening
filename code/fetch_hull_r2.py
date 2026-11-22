#!/usr/bin/env python3
"""Perluas data hull MP untuk PUTARAN 2 (lanjutan fetch_hull_subsystems.py).

Mengapa: hull 1.908 entri hanya mencakup 16 unsur kandidat putaran 1
(Ba, Bi, C, Ca, Eu, F, H, I, K, Li, Na, O, P, Si, Sr, Zn). Kandidat
putaran 2 memakai 16 unsur TAMBAHAN (Al, B, Ce, Cl, Cs, Dy, Hf, In, La,
Mg, Mn, Nd, S, Se, Sm, Tb) — tanpa entri subsistemnya, e_above_hull
mereka tidak bisa dihitung (r2_16 sudah terhitung energi formasinya
tapi tanpa vonis stabil).

Membaca 30 kandidat dari putaran2_draf30_v2.json (field `formula`),
menghitung semua subsistem tak-kosong tiap kandidat, lalu mengambil
yang belum tercakup dari API MP (chemsys eksak) dan MENAMBAHKAN ke
data_raw/mp_hull_subsystems.json yang sama (resume-aware: subsistem
yang sudah ada dilewati, entri didedup per material_id).

Kunci dari ~/.secrets/mp_api_key (tidak pernah dicetak/ditulis).
Jalankan: .venv-wp2/bin/python fetch_hull_r2.py
"""
import json, os, time
from itertools import combinations
import requests
from pymatgen.core import Composition

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
    cands = json.load(open(os.path.join(BASE_DIR, "putaran2_draf30_v2.json")))
    subsystems = set()
    new_els = set()
    for c in cands:
        els = sorted(str(e) for e in Composition(c["formula"]).elements)
        new_els.update(els)
        for r in range(1, len(els) + 1):
            for sub in combinations(els, r):
                subsystems.add(sub)
    subsystems = sorted(subsystems, key=lambda s: (len(s), s))
    print(f"Kandidat putaran 2: {len(cands)}; unsur: {sorted(new_els)}", flush=True)
    print(f"Subsistem unik: {len(subsystems)}", flush=True)

    out_path = os.path.join(OUT_DIR, "mp_hull_subsystems.json")
    seen, rows = set(), []
    if os.path.exists(out_path):
        rows = json.load(open(out_path))
        seen = {d.get("material_id") for d in rows}
        print(f"  melanjutkan: {len(rows)} entri sudah ada", flush=True)
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
