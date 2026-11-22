#!/usr/bin/env python3
"""Gabung skrining R2 (MP+COD) + NOMAD -> tier A/B gabungan, dedup per rumus.

Baca screening_r2.csv + screening_r2_nomad.csv (lokal, tanpa torch).
Aturan SAMA dengan verify_r2.py:
  - eligible: source COD/NOMAD, flag == "" (bukan rumus putaran 1)
  - filter bio: tanpa U/Th/Pa/Np/Pu/Am/Cm/Bk/Cf/Es/Fm/Md/No/Lr/Ac/Ra/Rn/Po/At/Fr/Tc/Pb/Cd/Hg/As/Tl
  - Tier A: Ca/P 1,50-1,67 ; Tier B: Ca/P 1,00-2,00
  - DEDUP per reduced_formula (simpan LCB terbaik) — aturan diversitas yang
    hilang di tierAB pertama (Ca5Cl2O12P3 muncul 5x, Ca5O10P3Sr4 3x).
Keluaran: screening_r2_merged_tierAB.json + ringkasan di stdout.
"""
import csv, json, re
from collections import Counter

BASE = "/home/hatch/workspace/research/PRGF baru"
BAD = {"U", "Th", "Pa", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
       "Md", "No", "Lr", "Ac", "Ra", "Rn", "Po", "At", "Fr", "Tc",
       "Pb", "Cd", "Hg", "As", "Tl"}
ELRE = re.compile(r"([A-Z][a-z]?)")

def els(f):
    return set(ELRE.findall(f))

def _count(m, el):
    # jumlah unsur: string kosong '' = 1 (implisit), tak ada = 0.
    # (m.get(el) or 0) SALAH untuk '' — '' or 0 = 0, padahal maksudnya 1.
    v = m.get(el)
    if v is None:
        return 0.0
    return float(v) if v else 1.0

def ca_p(f):
    m = dict(re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", f))
    ca = _count(m, "Ca")
    p = _count(m, "P")
    return round(ca / p, 2) if p else None

def tier(r):
    cp = ca_p(r["reduced_formula"])
    if cp is None:
        return "tanpa-P?"
    if 1.50 <= cp <= 1.67:
        return "A"
    if 1.00 <= cp <= 2.00:
        return "B"
    return "luar"

def load(name):
    return list(csv.DictReader(open(f"{BASE}/{name}")))

r2 = load("screening_r2.csv")
rn = load("screening_r2_nomad.csv")
all_rows = r2 + rn

uniq_old = {r["reduced_formula"] for r in r2}
uniq_new = {r["reduced_formula"] for r in rn}
overlap = uniq_old & uniq_new
uniq_all = uniq_old | uniq_new
print(f"Rumus unik: lama {len(uniq_old)} + NOMAD {len(uniq_new)} - irisan {len(overlap)} = {len(uniq_all)}")
print(f"Cakupan terhadap 3.249 rumus: {100*len(uniq_all)/3249:.1f}% (sebelumnya {100*len(uniq_old)/3249:.1f}%)")
if overlap:
    print("  irisan (contoh):", sorted(overlap)[:10])

elig = [r for r in all_rows if r["source"] in ("COD", "NOMAD") and r["flag"] == ""]
print(f"\nEligible (COD+NOMAD, bukan r1): {len(elig)} (COD {sum(1 for r in elig if r['source']=='COD')}, NOMAD {sum(1 for r in elig if r['source']=='NOMAD')})")
filt = [r for r in elig if not (els(r["reduced_formula"]) & BAD)]
print(f"Setelah filter bio: {len(filt)} (COD {sum(1 for r in filt if r['source']=='COD')}, NOMAD {sum(1 for r in filt if r['source']=='NOMAD')})")
print("Tier di eligible terfilter:", dict(Counter(tier(r) for r in filt)))

# dedup per rumus (LCB terbaik)
best = {}
for r in filt:
    k = r["reduced_formula"]
    if k not in best or float(r["lcb"]) < float(best[k]["lcb"]):
        best[k] = r
ded = list(best.values())
print(f"Setelah dedup per rumus: {len(ded)} (COD {sum(1 for r in ded if r['source']=='COD')}, NOMAD {sum(1 for r in ded if r['source']=='NOMAD')})")
print("Tier sesudah dedup:", dict(Counter(tier(r) for r in ded)))

tierA = sorted([r for r in ded if tier(r) == "A"], key=lambda x: float(x["lcb"]))[:15]
tierB = sorted([r for r in ded if tier(r) == "B"], key=lambda x: float(x["lcb"]))[:15]

def show(nama, lst):
    print(f"\n{nama} (dedup, filter bio):")
    for i, r in enumerate(lst, 1):
        no_o = "TANPA-O" if "O" not in els(r["reduced_formula"]) else ""
        hsd = "sd-tinggi" if float(r["sd"]) > 0.5 else ""
        print(f"{i:2d}. {r['reduced_formula']:28s} {r['source']:5s} {r['source_id']:32s} ns={r['nsites']:>4s} "
              f"Ca/P={ca_p(r['reduced_formula'])} mean={float(r['mean_eform']):+.4f} sd={float(r['sd']):.4f} "
              f"LCB={float(r['lcb']):+.4f} {no_o} {hsd}")

show("TIER A TOP 15 GABUNGAN", tierA)
show("TIER B TOP 15 GABUNGAN", tierB)

# bandingkan dengan tierAB lama (dedup dengan aturan yang sama, dari screening_r2.csv saja)
old_elig = [r for r in r2 if r["source"] == "COD" and r["flag"] == ""]
old_filt = [r for r in old_elig if not (els(r["reduced_formula"]) & BAD)]
old_best = {}
for r in old_filt:
    k = r["reduced_formula"]
    if k not in old_best or float(r["lcb"]) < float(old_best[k]["lcb"]):
        old_best[k] = r
old_ded = list(old_best.values())
oldA = sorted([r for r in old_ded if tier(r) == "A"], key=lambda x: float(x["lcb"]))[:15]
oldB = sorted([r for r in old_ded if tier(r) == "B"], key=lambda x: float(x["lcb"]))[:15]
oldA_f = {r["reduced_formula"] for r in oldA}
oldB_f = {r["reduced_formula"] for r in oldB}
newA_f = {r["reduced_formula"] for r in tierA}
newB_f = {r["reduced_formula"] for r in tierB}
print("\nPerbandingan vs Tier lama (aturan sama, tanpa NOMAD):")
print(f"  Tier A: {len(newA_f & oldA_f)} rumus bertahan, {len(newA_f - oldA_f)} baru, {len(oldA_f - newA_f)} terdepak")
print(f"    baru dari NOMAD di Tier A: {[r['reduced_formula'] for r in tierA if r['source']=='NOMAD']}")
print(f"    terdepak dari Tier A: {sorted(oldA_f - newA_f)}")
print(f"  Tier B: {len(newB_f & oldB_f)} rumus bertahan, {len(newB_f - oldB_f)} baru, {len(oldB_f - newB_f)} terdepak")
print(f"    baru dari NOMAD di Tier B: {[r['reduced_formula'] for r in tierB if r['source']=='NOMAD']}")
print(f"    terdepak dari Tier B: {sorted(oldB_f - newB_f)}")

out = {
    "coverage": {"unique_formulas_screened": len(uniq_all), "of_3249_pct": round(100*len(uniq_all)/3249, 1),
                 "old_unique": len(uniq_old), "nomad_unique": len(uniq_new), "overlap": len(overlap)},
    "eligible": {"total": len(elig), "bio_filtered": len(filt), "deduped": len(ded),
                 "tier_counts_deduped": dict(Counter(tier(r) for r in ded))},
    "tierA_top15": tierA,
    "tierB_top15": tierB,
    "comparison_vs_old": {
        "tierA_kept": sorted(newA_f & oldA_f), "tierA_new": sorted(newA_f - oldA_f),
        "tierA_displaced": sorted(oldA_f - newA_f),
        "tierB_kept": sorted(newB_f & oldB_f), "tierB_new": sorted(newB_f - oldB_f),
        "tierB_displaced": sorted(oldB_f - newB_f)},
    "notes": ["Dedup per reduced_formula (LCB terbaik); filter bio = verify_r2.py.",
              "TANPA-O = rumus tanpa oksigen (bukan fosfat — ragu sebagai scaffold tulang).",
              "sd-tinggi = sd > 0,5: LCB digerakkan ketidakpastian, bukan rerata."],
}
json.dump(out, open(f"{BASE}/screening_r2_merged_tierAB.json", "w"), indent=2)
print("\nDitulis: screening_r2_merged_tierAB.json")
