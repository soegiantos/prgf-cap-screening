#!/usr/bin/env python3
"""Hitung + BEKUKAN pilihan aturan pembanding R6 / R6b — PRGF baru.

Uji prospektif (rancangan teman pemilik, 2026-11-22): R6 = bobot lunak
0,20 top-3 dari kolam R2 (9 kandidat); R6b = S_stab x faktor biologi
(1,0/0,5/0,1) top-3, eksploratif. H1: presisi@3 R5 >= presisi@3 R6
pada label prospektif. H2: r2_26 tidak stabil (E_hull > 0,05).

PENTING — dua verifikasi dari data (bukan asumsi):
(1) Kolam R2 dari aturan_seleksi_v2.json = 9 kandidat (bukan 25):
    r2_04, r2_15, r2_18, r2_21, r2_25, r2_26, r2_27, r2_29, r2_30.
    Primer dihitung dari kolam 9; sensitivitas dari kolam 25 tak berlabel.
(2) "Skor peringkat awal ronde 2" TIDAK ADA sebagai artefak beku: bobot
    0,20 tidak pernah diimplementasikan di putaran 2 (rank di manifest =
    urutan LCB mentah dalam tier). R6/R6b adalah REKONSTRUKSI, dilabeli
    begitu di berkas pembekuan.

Definisi (dibekukan sebelum verdict apa pun):
  bio    = min skor ion atas semua unsur asing (di luar Ca/P/O/H):
           2 untuk Zn/Sr/Mg/Si; 1 untuk Mn/F/Na/K/Cl; 0 untuk lainnya.
           Senyawa Ca-P murni (tanpa unsur asing) = 2.
  S_stab = normalisasi min-max hull_LCB atas kolam yang diranking:
           (max(hl) - hl) / (max(hl) - min(hl)) -> 1 = paling stabil.
  R6  : S = 0,80*S_stab + 0,20*(bio/2). Top-3.
  R6b : S = S_stab * faktor; faktor = 1,0 bila tanpa ion asing atau
        semua ion asing skor 2; 0,5 bila ada ion skor 1 dan tanpa ion
        skor 0; 0,1 bila ada ion tak terverifikasi (skor 0). Top-3.
Tidak mengubah aturan_seleksi_v2.py/.json, manuscript.*, atau
soft_weight_backtest.py; hanya membaca aturan_seleksi_v2.json.
"""
import json
from pymatgen.core import Composition

BASE = "/home/hatch/workspace/research/PRGF baru"
d = json.load(open(f"{BASE}/aturan_seleksi_v2.json"))
rows = d["kandidat"]

SKOR_ION = {"Zn": 2, "Sr": 2, "Mg": 2, "Si": 2,
            "Mn": 1, "F": 1, "Na": 1, "K": 1, "Cl": 1}
DASAR = {"Ca", "P", "O", "H"}


def bio_score(formula):
    els = {str(x) for x in Composition(formula).elements}
    asing = els - DASAR
    if not asing:
        return 2
    return min(SKOR_ION.get(e, 0) for e in asing)


def faktor_bio(formula):
    els = {str(x) for x in Composition(formula).elements}
    asing = els - DASAR
    if not asing:
        return 1.0
    skors = [SKOR_ION.get(e, 0) for e in asing]
    if min(skors) == 0:
        return 0.1
    if min(skors) == 1:
        return 0.5
    return 1.0


for r in rows:
    r["bio"] = bio_score(r["formula"])
    r["faktor"] = faktor_bio(r["formula"])

lab = [r for r in rows if "stabil_dft" in r]
unl = [r for r in rows if "stabil_dft" not in r]
assert len(lab) == 5 and len(unl) == 25, (len(lab), len(unl))

pool9 = [r for r in rows if r["job"] in d["aturan"]["R2"]["sisa"]]
assert len(pool9) == 9, [r["job"] for r in pool9]
print("Kolam R2 (primer, 9) terverifikasi dari JSON:")
print(" ", [r["job"] for r in pool9])
print("Kolam 25 tak berlabel:", len(unl), "kandidat")


def skor_kolam(pool, norm_pool):
    hls = [r["hull_lcb"] for r in norm_pool]
    lo, hi = min(hls), max(hls)
    out = []
    for r in pool:
        s_stab = (hi - r["hull_lcb"]) / (hi - lo)
        s6 = 0.80 * s_stab + 0.20 * (r["bio"] / 2)
        s6b = s_stab * r["faktor"]
        out.append({**r, "s_stab": s_stab, "S6": s6, "S6b": s6b})
    return out, lo, hi


def top3(pool_skor, kunci):
    return sorted(pool_skor, key=lambda x: (-x[kunci], x["job"]))[:3]


def cetak_tabel(pool_skor, kunci, judul):
    print(f"\n== {judul} ==")
    print(f"  {'job':28s} {'hl':>7s} {'bio':>3s} {'fak':>4s} "
          f"{'S_stab':>7s} {kunci:>6s}")
    for r in sorted(pool_skor, key=lambda x: -x[kunci]):
        print(f"  {r['job']:28s} {r['hull_lcb']:+7.4f} {r['bio']:3d} "
              f"{r['faktor']:4.1f} {r['s_stab']:7.3f} {r[kunci]:6.3f}")


def backtest_retro(pilih, nama):
    tp = sum(1 for r in lab if pilih(r) and r["stabil_dft"])
    fp = sum(1 for r in lab if pilih(r) and not r["stabil_dft"])
    fn = sum(1 for r in lab if not pilih(r) and r["stabil_dft"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    print(f"{nama}: TP={tp} FP={fp} FN={fn} presisi={prec} recall={rec}")
    return {"TP": tp, "FP": fp, "FN": fn, "presisi": prec, "recall": rec}


# ---------- R6 primer: kolam 9, normalisasi atas kolam 9 ----------
p9, lo9, hi9 = skor_kolam(pool9, pool9)
print(f"\nNormalisasi primer kolam 9: hull_LCB min {lo9:+.4f} max {hi9:+.4f}")
cetak_tabel(p9, "S6", "R6 PRIMER — kolam 9, norm kolam 9")
cetak_tabel(p9, "S6b", "R6b PRIMER — kolam 9, norm kolam 9")
r6_primer = [r["job"] for r in top3(p9, "S6")]
r6b_primer = [r["job"] for r in top3(p9, "S6b")]
print("\nR6  primer top-3:", r6_primer)
print("R6b primer top-3:", r6b_primer)

# ---------- R6 sensitivitas A: kolam 9, normalisasi atas 30 draf ----------
p9n30, lo30, hi30 = skor_kolam(pool9, rows)
print(f"\nNormalisasi sensitivitas: 30 draf hull_LCB min {lo30:+.4f} "
      f"max {hi30:+.4f}")
cetak_tabel(p9n30, "S6", "R6 SENS-A — kolam 9, norm 30 draf")
r6_sensA = [r["job"] for r in top3(p9n30, "S6")]
print("R6 sens-A top-3:", r6_sensA)

# ---------- R6 + R6b sensitivitas B: kolam 25, normalisasi atas kolam 25 ----------
p25, lo25, hi25 = skor_kolam(unl, unl)
print(f"\nNormalisasi kolam 25: hull_LCB min {lo25:+.4f} max {hi25:+.4f}")
cetak_tabel(p25, "S6", "R6 SENS-B — kolam 25, norm kolam 25")
cetak_tabel(p25, "S6b", "R6b SENS — kolam 25, norm kolam 25")
r6_sensB = [r["job"] for r in top3(p25, "S6")]
r6b_sens = [r["job"] for r in top3(p25, "S6b")]
print("\nR6 sens-B top-3:", r6_sensB)
print("R6b sens  top-3:", r6b_sens)

# ---------- Retrospektif pada 5 label (norm 30, top-3 atas 30 draf) ----------
# Konvensi sama dengan soft_weight_backtest.py (SW top-3).
p30, _, _ = skor_kolam(rows, rows)
top3_6 = {r["job"] for r in top3(p30, "S6")}
top3_6b = {r["job"] for r in top3(p30, "S6b")}
print("\n== RETROSPEKTIF (norm 30, top-3 atas 30 draf; "
      "TIDAK dipakai untuk keputusan) ==")
print("R6  retro top-3:", sorted(top3_6))
print("R6b retro top-3:", sorted(top3_6b))
backtest_retro(lambda r: r["job"] in top3_6, "R6  retro")
backtest_retro(lambda r: r["job"] in top3_6b, "R6b retro")
for rule in ("R2", "R5"):
    a = d["aturan"][rule]
    print(f"{rule}: TP={a['TP']} FP={a['FP']} FN={a['FN']} | "
          f"memilih {a['dipilih_dari_sisa_25']}/25")

# ---------- Clopper-Pearson 95% untuk presisi@3 bila 3/3 stabil ----------
try:
    from scipy.stats import beta as _beta
    lo_cp = _beta.ppf(0.025, 3, 1)
    hi_cp = _beta.ppf(0.975, 4, 0) if False else 1.0
    print(f"\nClopper-Pearson 95% untuk 3/3 stabil (scipy): "
          f"[{lo_cp:.3f}, {hi_cp:.3f}]")
except Exception as e:
    lo_cp = 0.025 ** (1 / 3)
    print(f"\nClopper-Pearson 95% untuk 3/3 stabil (rumus eksak beta, "
          f"tanpa scipy): [{lo_cp:.3f}, 1.000]")

# ---------- Ringkasan untuk berkas pembekuan ----------
ringkas = {
    "pool9": [r["job"] for r in pool9],
    "R5": d["aturan"]["R5"]["sisa"],
    "R6_primer": r6_primer,
    "R6b_primer": r6b_primer,
    "R6_sensA_norm30": r6_sensA,
    "R6_sensB_pool25": r6_sensB,
    "R6b_sens_pool25": r6b_sens,
    "R6_retro_top3_30": sorted(top3_6),
    "R6b_retro_top3_30": sorted(top3_6b),
    "CP95_3dari3": [round(lo_cp, 3), 1.0],
}
print("\n== RINGKASAN ==")
print(json.dumps(ringkas, indent=2))
json.dump(ringkas, open(f"{BASE}/r6_ringkas.json", "w"), indent=2)
print("\nRingkasan tersimpan di r6_ringkas.json")
