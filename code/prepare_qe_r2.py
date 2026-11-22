#!/usr/bin/env python3
"""Siapkan input Quantum ESPRESSO (vc-relax) untuk 30 kandidat DFT WP3 putaran 2 (draf v2).

Setting SAMA PERSIS dengan putaran 1 (prepare_qe_wp3.py) dan pipeline DFT
produksi PRGF: PBE, pseudo PSL 1.0.0, ecutwfc 80 / ecutrho 640 Ry, smearing
gaussian 0.01 Ry, conv_thr 1e-10, mixing_beta 0.7, forc_conv_thr 1e-4 (BFGS),
tstress + tprnfor, ibrav 0; k-mesh 3x3x3 (<=60 situs), 2x2x2 (<=150), 1x1x1 (>150).
Struktur per sumber:
  - MP (14): dari wp2_dataset.json mp_entries (struktur DFT terurut).
  - COD (11): parse CIF (data_raw/cod_cif) + pengisian mayoritas + UJI
    KEJUJURAN komposisi seperti putaran 1 (rasio terurut/terparse per unsur
    ~sama dalam 5%, tak ada unsur hilang) — yang gagal DITUNDA ke daftar
    butuh-supercell-terurut (pola putaran 1b), bukan dipaksakan.
  - NOMAD (5): dari data_raw/nomad_structures.jsonl (kartesian -> fraksional);
    struktur DFT terurut, tanpa okupansi parsial — dicek dmin antar atom.
Pseudo: 16 lama (dft_r1/pseudo/) + 7 baru (Ce, Dy, La, Mg, Nd, Sm, Tb —
unduhan PSL 1.0.0, semua *-spn-kjpaw seperti Eu/Ba).
Jalankan: .venv-wp2/bin/python prepare_qe_r2.py
Keluaran: dft_r2/*.in, dft_r2/pseudo/*.UPF, dft_r2/run_all.sh (tahan gagal
per job, lanjut ke berikut), dft_r2/manifest.json. HANYA MENYIAPKAN.
"""
import json, os, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pymatgen.core import Element, Structure, Lattice
from pymatgen.io.cif import CifParser

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dft_r2")
PSEUDO_DIR = os.path.join(OUT, "pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

PSEUDO = {
    "Al": "Al.pbe-nl-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Bi": "Bi.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "C": "C.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ce": "Ce.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Cl": "Cl.pbe-nl-kjpaw_psl.1.0.0.UPF",
    "Cs": "Cs.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Dy": "Dy.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Eu": "Eu.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "F": "F.pbe-n-rrkjus_psl.1.0.0.UPF",
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
    "Hf": "Hf.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "I": "I.pbe-n-rrkjus_psl.1.0.0.UPF",
    "In": "In.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "La": "La.pbe-spfn-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF",
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Mn": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Nd": "Nd.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S": "S.pbe-nl-kjpaw_psl.1.0.0.UPF",
    "Se": "Se.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Sm": "Sm.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Sr": "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Tb": "Tb.pbe-spdn-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF",
}

def k_mesh(nsites):
    if nsites <= 60:
        return (3, 3, 3)
    if nsites <= 150:
        return (2, 2, 2)
    return (1, 1, 1)

def order_structure(struct):
    """Tiap situs -> spesies mayoritas. Kembalikan (struktur_terurut, perubahan)."""
    species, coords, changes = [], [], []
    for site in struct:
        comp = site.species.as_dict()
        top = max(comp, key=comp.get)
        if len(comp) > 1 or abs(comp[top] - 1.0) > 0.02:
            changes.append({str(site.species): comp})
        species.append(top)
        coords.append(site.frac_coords)
    s = Structure(struct.lattice, species, coords, coords_are_cartesian=False)
    return s, changes

def dmin_structure(struct):
    """Jarak antar atom minimum (PBC, min-image kasar via 27 shift)."""
    lat = np.array(struct.lattice.matrix, float)
    frac = np.array([s.frac_coords for s in struct], float)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    n = len(frac)
    dmin = np.inf
    for a in range(n):
        diff = frac[None, :, :] + shifts[:, None, :] - frac[a]
        dist = np.linalg.norm(diff @ lat, axis=-1)
        dist[13, a] = np.inf
        dmin = min(dmin, float(dist.min()))
    return dmin

def write_qe(job, struct, kpts):
    els = sorted({site.specie.symbol for site in struct}, key=lambda e: Element(e).Z)
    nat = len(struct)
    lines = []
    lines.append("&CONTROL")
    lines.append(f"   calculation      = 'vc-relax'")
    lines.append("   verbosity        = 'high'")
    lines.append("   tstress          = .true.")
    lines.append("   tprnfor          = .true.")
    lines.append(f"   outdir           = './tmp_{job}'")
    lines.append(f"   prefix           = '{job}'")
    lines.append("   etot_conv_thr    = 1e-07")
    lines.append("   forc_conv_thr    = 0.0001")
    lines.append("   pseudo_dir       = './pseudo'")
    lines.append("/")
    lines.append("&SYSTEM")
    lines.append("   ecutwfc          = 80")
    lines.append("   ecutrho          = 640")
    lines.append("   degauss          = 0.01")
    lines.append("   smearing         = 'gaussian'")
    lines.append(f"   ntyp             = {len(els)}")
    lines.append(f"   nat              = {nat}")
    lines.append("   ibrav            = 0")
    lines.append("/")
    lines.append("&ELECTRONS")
    lines.append("   conv_thr         = 1e-10")
    lines.append("   mixing_beta      = 0.7")
    lines.append("/")
    lines.append("&IONS")
    lines.append("   ion_dynamics     = 'bfgs'")
    lines.append("/")
    lines.append("&CELL")
    lines.append("   cell_dynamics    = 'bfgs'")
    lines.append("   press_conv_thr   = 0.5")
    lines.append("/")
    lines.append("ATOMIC_SPECIES")
    for el in els:
        lines.append(f"{el} {Element(el).atomic_mass:.6f} {PSEUDO[el]}")
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"{kpts[0]} {kpts[1]} {kpts[2]}  0 0 0")
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for v in struct.lattice.matrix:
        lines.append(" ".join(f"{x:.10f}" for x in v))
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in struct:
        f = site.frac_coords
        lines.append(f"{site.specie.symbol} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}")
    return "\n".join(lines) + "\n"

def formula_counts(f):
    """Rumus -> {unsur: jumlah}. 'Ca3(PO4)2' pymatgen == 'Ca3O8P2'."""
    import re as _re
    def parse(s, mult=1.0, out=None):
        out = {} if out is None else out
        i = 0
        while i < len(s):
            if s[i] == "(":
                j = s.index(")", i)
                sub = parse(s[i + 1:j], 1.0, {})
                k = j + 1
                num = ""
                while k < len(s) and (s[k].isdigit() or s[k] == "."):
                    num += s[k]; k += 1
                m = float(num) if num else 1.0
                for el, v in sub.items():
                    out[el] = out.get(el, 0.0) + v * m * mult
                i = k
            elif s[i].isupper():
                m = _re.match(r"[A-Z][a-z]?", s[i:])
                el = m.group(0)
                j = i + len(el)
                num = ""
                while j < len(s) and (s[j].isdigit() or s[j] == "."):
                    num += s[j]; j += 1
                out[el] = out.get(el, 0.0) + (float(num) if num else 1.0) * mult
                i = j
            else:
                i += 1
        return out
    return parse(f)

def same_composition(struct, formula):
    """Bandingkan komposisi lewat RASIO (struktur MP bisa berisi kelipatan
    satuan rumus, mis. Ca10P6O24F2 = 2x Ca5FO12P3)."""
    a = struct.composition.as_dict()
    b = formula_counts(formula)
    if set(a) != set(b):
        return False
    na = min(a.values()); nb = min(b.values())
    return all(abs(a[e] / na - b[e] / nb) < 0.01 for e in a)

# ---------- muat draf v2 + data struktur ----------
draft = json.load(open(f"{BASE}/putaran2_draf30_v2.json"))
ds = json.load(open(f"{BASE}/wp2_dataset.json"))
mp_by_id = {e["material_id"]: e for e in ds["mp_entries"]}
cod_by_id = {e["file_id"]: e for e in ds["cod_entries"]}
nomad_by_id = {}
for line in open(f"{BASE}/data_raw/nomad_structures.jsonl"):
    r = json.loads(line)
    nomad_by_id[r["entry_id"]] = r

# salin pseudo lama (16) dari dft_r1/pseudo/
for fn in set(PSEUDO.values()):
    dst = os.path.join(PSEUDO_DIR, fn)
    if os.path.exists(dst):
        continue
    src = os.path.join(BASE, "dft_r1", "pseudo", fn)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"Pseudo belum ada (unduh dulu): {fn}")

missing_upf = [fn for fn in set(PSEUDO.values())
               if not os.path.exists(os.path.join(PSEUDO_DIR, fn))]
if missing_upf:
    raise SystemExit(f"Pseudo hilang: {missing_upf} — unduh dari PSL dulu.")

# ---------- bangun input ----------
jobs, deferred = [], []
for i, c in enumerate(draft, 1):
    job = f"r2_{i:02d}_{c['formula']}"
    src, sid = c["source"], c["source_id"]
    changes, dmin, note = [], None, ""
    if src == "MP":
        e = mp_by_id[sid]
        struct = Structure(Lattice(e["lattice"]), e["elements"], e["frac_coords"],
                           coords_are_cartesian=False)
        assert same_composition(struct, c["formula"]), \
            f"{job}: rumus MP {struct.composition.reduced_formula} != draf {c['formula']}"
        dmin = dmin_structure(struct)
        note = "struktur MP (terurut, DFT)"
    elif src == "COD":
        fid = sid.replace("COD-", "")
        # occupancy_tolerance=1.2: beberapa CIF COD okupansinya >1 (sloppy,
        # mis. 2022591 sampai 1,153) — toleransi default 1.0 menolak parse.
        s = CifParser(f"{BASE}/data_raw/cod_cif/{fid}.cif",
                      occupancy_tolerance=1.2).parse_structures(primitive=False)[0]
        struct, changes = order_structure(s)
        parsed = s.composition.as_dict()
        ordered = struct.composition.as_dict()
        els_all = set(parsed) | set(ordered)
        ratios, ok_ratio = [], True
        for el in els_all:
            if el not in parsed or el not in ordered:
                ok_ratio = False
                break
            ratios.append(ordered[el] / parsed[el])
        if ok_ratio and ratios and (max(ratios) / min(ratios) > 1.05):
            ok_ratio = False
        if not ok_ratio:
            deferred.append({
                "job": job, "rank": i, "tier": c["tier"], "formula": c["formula"],
                "source": "COD", "source_id": sid, "nsites": len(struct),
                "formula_terparse": s.composition.reduced_formula,
                "formula_terurut": struct.composition.reduced_formula,
                "alasan": ("pengisian mayoritas mengubah komposisi: "
                           f"terparse {s.composition.reduced_formula} -> terurut {struct.composition.reduced_formula}; "
                           "butuh supercell terurut pola putaran 1b"),
            })
            print(f"{job}: DITUNDA — komposisi berubah "
                  f"{s.composition.reduced_formula} -> {struct.composition.reduced_formula}")
            continue
        dmin = dmin_structure(struct)
        note = f"COD {fid} (terurut mayoritas; {len(changes)} situs okupansi diubah)" if changes else f"COD {fid} (bersih)"
    elif src == "NOMAD":
        eid = sid.replace("NOMAD-", "")
        r = nomad_by_id[eid]
        els = ["Ca" if x == "Ca_sv" else x for x in r["labels"]]
        lat = np.array(r["lattice_vectors"], float)
        frac = (np.array(r["positions"], float) @ np.linalg.inv(lat)) % 1.0
        struct = Structure(Lattice(lat), els, frac, coords_are_cartesian=False)
        assert same_composition(struct, c["formula"]), \
            f"{job}: rumus NOMAD {struct.composition.reduced_formula} != draf {c['formula']}"
        dmin = dmin_structure(struct)
        note = f"NOMAD {eid} (terurut, DFT)"
    else:
        raise SystemExit(f"sumber tak dikenal: {src}")
    if dmin is not None and dmin < 1.0:
        print(f"  PERINGATAN {job}: dmin {dmin:.2f} Angstrom < 1,0 — periksa struktur!")
    kpts = k_mesh(len(struct))
    text = write_qe(job, struct, kpts)
    path = os.path.join(OUT, f"{job}.in")
    open(path, "w").write(text)
    els = sorted({site.specie.symbol for site in struct})
    jobs.append({
        "job": job, "file": f"{job}.in",
        "rank": i, "tier": c["tier"], "formula": c["formula"],
        "ca_p": c["ca_p"], "source": src, "source_id": sid,
        "nsites": len(struct), "k_mesh": list(kpts), "elements": els,
        "pseudo": [PSEUDO[e] for e in els],
        "lcb": c["lcb"], "sd": c["sd"], "mean_eform": c["mean_eform"],
        "mp_e_form": c["mp_e_form"], "dmin_A": round(dmin, 3) if dmin else None,
        "perubahan_okupansi": changes, "catatan": note,
    })
    print(f"{job}: {len(struct)} situs, k={kpts}, unsur={','.join(els)}, dmin={dmin:.2f} A — {note}")

# ---------- verifikasi ----------
print("\n== verifikasi ==")
ok = True
for j in jobs:
    txt = open(os.path.join(OUT, j["file"])).read()
    npos = len([l for l in txt.split("ATOMIC_POSITIONS crystal")[1].strip().split("\n") if l.strip()])
    if npos != j["nsites"]:
        print(f"  {j['job']}: JUMLAH POSISI {npos} != nsites {j['nsites']}"); ok = False
    if f"nat              = {j['nsites']}" not in txt:
        print(f"  {j['job']}: nat tidak cocok"); ok = False
print("Verifikasi:", f"OK — {len(jobs)} input konsisten" if ok else "ADA MASALAH (lihat di atas)")

# ---------- run order: situs kecil dulu, Tier A didahulukan bila seri ----------
order = sorted(jobs, key=lambda j: (j["nsites"], 0 if j["tier"] == "A" else 1))
for k, j in enumerate(order, 1):
    j["urutan_jalan"] = k
run_sh = ["#!/bin/bash",
          "# Jalankan DFT WP3 putaran 2 (draf v2) — SATU PER SATU, urutan situs kecil dulu.",
          "# Tahan gagal per job (lanjut ke berikut bila pw.x exit!=0).",
          "# JANGAN jalankan bersamaan dengan job DFT lain di mesin yang sama.",
          "# Contoh: nohup bash run_all.sh > run_r2.log 2>&1 &",
          "PWX=${PWX:-/root/miniconda3/bin/pw.x}",
          ""]
for j in order:
    run_sh.append(f'echo "== {j["urutan_jalan"]}/{len(order)} {j["job"]} =="')
    run_sh.append(f'"$PWX" -in {j["file"]} > {j["job"]}.out 2>&1 || echo "PWX exit!=0 untuk {j["job"]} — lanjut ke job berikut"')
    run_sh.append(f'grep -q "JOB DONE" {j["job"]}.out && echo OK || echo "GAGAL — cek {j["job"]}.out"')
open(os.path.join(OUT, "run_all.sh"), "w").write("\n".join(run_sh) + "\n")

manifest = {
    "protokol": "vc-relax PBE + PSL 1.0.0, ecut 80/640 Ry, smearing gaussian 0.01, conv_thr 1e-10, mixing 0.7, forc 1e-4; k-mesh 3x3x3 (<=60 situs), 2x2x2 (<=150), 1x1x1 (>150)",
    "sumber_setting": "sama dengan putaran 1 (prepare_qe_wp3.py) + pipeline DFT produksi PRGF",
    "draf": "putaran2_draf30_v2.json (revisi 'lakukan yang terbaik': syarat O Tier B, 5 NOMAD masuk via LCB ketat, mulai saat mesin luang)",
    "status": (f"SIAP — {len(jobs)} input bersih + {len(deferred)} ditunda (butuh supercell terurut pola 1b). "
               "BELUM dijalankan: putaran 1 masih jalan di 3 mesin. Luncur per mesin begitu luang "
               "(VPS dulu sesudah ref selesai; lalu Mac dan SOEGI-LAB)."),
    "n_siap": len(jobs),
    "n_ditunda": len(deferred),
    "jobs": jobs,
    "ditunda_butuh_supercell_terurut": deferred,
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(f"\nWrote dft_r2/: {len(jobs)} .in siap + {len(deferred)} ditunda + pseudo/ (23 UPF) + run_all.sh + manifest.json")
print("Urutan jalan (situs kecil dulu):")
for j in order:
    print(f"  {j['urutan_jalan']:2d}. {j['job']} ({j['nsites']} situs, k={j['k_mesh']}, {j['source']})")
