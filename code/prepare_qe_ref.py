#!/usr/bin/env python3
"""Siapkan input QE untuk energi referensi unsur (mu_QE) — WAJIB sebelum
energi formasi kandidat WP3 bisa dihitung dari energi total QE.

Alasan (terbukti oleh --calibrate di analisis_dft_r1.py): energi total absolut
QE dan VASP/MP tidak sebanding (pseudopotential berbeda) — referensi unsur
harus dihitung dengan KODE dan PSEUDO YANG SAMA (QE + PSL 1.0.0) seperti
kandidatnya. Referensi yang dipakai (keadaan standar, 0 K):
  H2, O2, F2 : molekul dalam kotak kubus 12 A (O2 triplet: nspin=2, mag=2)
  Ca fcc, K bcc, Li bcc, Ba bcc, Zn hcp, P hitam (Cmca), C grafit (P63/mmc)
Konstanta kisi awal dari literatur (vc-relax akan mengoptimalkannya, jadi
angka awal tidak kritis). Setting SAMA dengan pipeline (ecut 80/640,
smearing gaussian 0.01, conv_thr 1e-10, mixing 0.7, forc 1e-4).
Molekul: calculation='relax' (sel tetap) — vc-relax akan meruntuhkan kotak.
k-mesh: logam kecil 8x8x8; P hitam 6x2x5; grafit 8x8x4; molekul Gamma.

Jalankan: .venv-wp2/bin/python prepare_qe_ref.py
Keluaran: dft_ref/*.in, dft_ref/pseudo/*.UPF, dft_ref/run_ref.sh,
dft_ref/manifest.json. HANYA MENYIAPKAN — tidak menjalankan DFT.
Setelah job selesai, jalankan analisis_dft_r1.py --mu-qe dft_ref
untuk menulis dft_r1/ref_qe_mu.json.
"""
import json, os, shutil, warnings
warnings.filterwarnings("ignore")
from pymatgen.core import Element, Structure, Lattice

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dft_ref")
PSEUDO_SRC = os.path.join(BASE, "dft_r1", "pseudo")
PSEUDO_DIR = os.path.join(OUT, "pseudo")
os.makedirs(PSEUDO_DIR, exist_ok=True)

PSEUDO = {
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "F": "F.pbe-n-rrkjus_psl.1.0.0.UPF",
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
    "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF",
}

def molecule(el, d):
    lat = Lattice.cubic(12.0)
    return Structure(lat, [el, el], [[0.5 - d / 24, 0.5, 0.5], [0.5 + d / 24, 0.5, 0.5]],
                     coords_are_cartesian=False)

def bcc(el, a):
    return Structure(Lattice.cubic(a), [el, el], [[0, 0, 0], [0.5, 0.5, 0.5]])

def fcc(el, a):
    return Structure(Lattice.cubic(a), [el] * 4,
                     [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])

REFS = [
    {"job": "ref_H2", "el": "H", "struct": molecule("H", 0.74), "calc": "relax", "k": (1, 1, 1),
     "note": "molekul H2, kotak 12 A"},
    {"job": "ref_O2", "el": "O", "struct": molecule("O", 1.21), "calc": "relax", "k": (1, 1, 1),
     "note": "molekul O2 triplet, kotak 12 A", "nspin": 2, "tot_mag": 2.0},
    {"job": "ref_F2", "el": "F", "struct": molecule("F", 1.41), "calc": "relax", "k": (1, 1, 1),
     "note": "molekul F2, kotak 12 A"},
    {"job": "ref_Ca", "el": "Ca", "struct": fcc("Ca", 5.58), "calc": "vc-relax", "k": (8, 8, 8),
     "note": "Ca fcc a=5.58 A"},
    {"job": "ref_K", "el": "K", "struct": bcc("K", 5.32), "calc": "vc-relax", "k": (8, 8, 8),
     "note": "K bcc a=5.32 A"},
    {"job": "ref_Li", "el": "Li", "struct": bcc("Li", 3.51), "calc": "vc-relax", "k": (8, 8, 8),
     "note": "Li bcc a=3.51 A"},
    {"job": "ref_Ba", "el": "Ba", "struct": bcc("Ba", 5.02), "calc": "vc-relax", "k": (8, 8, 8),
     "note": "Ba bcc a=5.02 A"},
    {"job": "ref_Zn", "el": "Zn",
     "struct": Structure(Lattice.hexagonal(2.665, 4.947), ["Zn", "Zn"],
                         [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]),
     "calc": "vc-relax", "k": (8, 8, 8), "note": "Zn hcp a=2.665 c=4.947 A"},
    {"job": "ref_P", "el": "P",
     "struct": Structure.from_spacegroup("Cmce", Lattice.orthorhombic(3.3136, 4.3763, 10.478),
                                         ["P"], [[0, 0.08056, 0.10168]]),
     "calc": "vc-relax", "k": (6, 5, 2), "note": "P hitam Cmce, 8 atom (jarak P-P terdekat 2.24 A, terverifikasi)"},
    {"job": "ref_C", "el": "C",
     "struct": Structure(Lattice.hexagonal(2.461, 6.708), ["C"] * 4,
                         [[0, 0, 0.25], [2 / 3, 1 / 3, 0.25], [0, 0, 0.75], [1 / 3, 2 / 3, 0.75]]),
     "calc": "vc-relax", "k": (8, 8, 4), "note": "C grafit P63/mmc, 4 atom"},
]

def write_qe(r):
    s = r["struct"]
    els = sorted({site.specie.symbol for site in s})
    lines = ["&CONTROL",
             f"   calculation      = '{r['calc']}'",
             "   verbosity        = 'high'",
             "   tstress          = .true.",
             "   tprnfor          = .true.",
             f"   outdir           = './tmp_{r['job']}'",
             f"   prefix           = '{r['job']}'",
             "   etot_conv_thr    = 1e-07",
             "   forc_conv_thr    = 0.0001",
             "   pseudo_dir       = './pseudo'",
             "/",
             "&SYSTEM",
             "   ecutwfc          = 80",
             "   ecutrho          = 640",
             "   degauss          = 0.01",
             "   smearing         = 'gaussian'"]
    if r.get("nspin"):
        lines.append(f"   nspin            = {r['nspin']}")
        lines.append(f"   tot_magnetization = {r['tot_mag']}")
    lines += [f"   ntyp             = {len(els)}",
              f"   nat              = {len(s)}",
              "   ibrav            = 0",
              "/",
              "&ELECTRONS",
              "   conv_thr         = 1e-10",
              "   mixing_beta      = 0.7",
              "/",
              "&IONS",
              "   ion_dynamics     = 'bfgs'",
              "/"]
    if r["calc"] == "vc-relax":
        lines += ["&CELL",
                  "   cell_dynamics    = 'bfgs'",
                  "   press_conv_thr   = 0.5",
                  "/"]
    lines.append("ATOMIC_SPECIES")
    for el in els:
        lines.append(f"{el} {Element(el).atomic_mass:.6f} {PSEUDO[el]}")
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append(f"{r['k'][0]} {r['k'][1]} {r['k'][2]}  0 0 0")
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    for v in s.lattice.matrix:
        lines.append(" ".join(f"{x:.10f}" for x in v))
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    for site in s:
        f = site.frac_coords
        lines.append(f"{site.specie.symbol} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}")
    return "\n".join(lines) + "\n"

jobs = []
for r in REFS:
    for el in {site.specie.symbol for site in r["struct"]}:
        shutil.copy(os.path.join(PSEUDO_SRC, PSEUDO[el]), os.path.join(PSEUDO_DIR, PSEUDO[el]))
    path = os.path.join(OUT, f"{r['job']}.in")
    open(path, "w").write(write_qe(r))
    jobs.append({"job": r["job"], "file": f"{r['job']}.in", "element": r["el"],
                 "nsites": len(r["struct"]), "calculation": r["calc"], "k_mesh": list(r["k"]),
                 "pseudo": PSEUDO[r["el"]], "note": r["note"]})
    print(f"{r['job']}: {len(r['struct'])} situs, {r['calc']}, k={r['k']} — {r['note']}")

run_sh = ["#!/bin/bash",
          "# Job referensi unsur QE — KECIL, tiap job hitungan menit.",
          "# Jalankan SETELAH e6_m selesai (boleh sebelum/sesudah 7 kandidat r1).",
          "# Jalankan dari dalam folder dft_ref.",
          "PWX=${PWX:-/root/miniconda3/bin/pw.x}",
          "set -e"]
for i, j in enumerate(jobs, 1):
    run_sh.append(f'echo "== {i}/{len(jobs)} {j["job"]} =="')
    run_sh.append(f'"$PWX" -in {j["file"]} > {j["job"]}.out 2>&1 || echo "PWX exit!=0 untuk {j["job"]} — lanjut"')
    run_sh.append(f'grep -q "JOB DONE" {j["job"]}.out && echo OK || echo "GAGAL — cek {j["job"]}.out"')
open(os.path.join(OUT, "run_ref.sh"), "w").write("\n".join(run_sh) + "\n")

manifest = {
    "protokol": "sama dengan pipeline DFT PRGF (PBE + PSL 1.0.0, ecut 80/640, smearing gaussian 0.01, conv_thr 1e-10, mixing 0.7, forc 1e-4)",
    "tujuan": "energi referensi unsur mu_QE agar energi formasi kandidat bisa dihitung dari energi total QE (energi total QE dan VASP tidak sebanding)",
    "status": "SIAP — belum dijalankan. Jalankan setelah e6_m selesai; butuh akses eksekusi pw.x (root).",
    "jobs": jobs,
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(f"\nWrote dft_ref/: {len(jobs)} .in + pseudo/ + run_ref.sh + manifest.json")
