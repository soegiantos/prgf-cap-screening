#!/usr/bin/env python3
"""Aturan seleksi v2 — PRGF baru (backtest di 5 label DFT putaran 2).

Latar: seleksi putaran 2 memakai LCB = mean - 1,85*sd (s dikalibrasi di
data uji MP). Hasilnya: 4 dari 5 yang dipilih ternyata TIDAK STABIL —
GNN melebih-lebihkan kestabilan struktur NOMAD (meleset +0,17..+1,09).
Skrip ini menghitung, untuk ke-30 kandidat putaran 2, jarak hull yang
TERSIRAT oleh prediksi GNN (mean dan LCB) terhadap hull MP yang sama
dengan yang dipakai analisis DFT — jadi angka GNN dan DFT bisa
dibandingkan apel-dengan-apel — lalu menguji beberapa aturan seleksi
terhadap 5 label DFT yang sudah ada (1 stabil = r2_01 beta-TCP,
4 tidak stabil = r2_16/17/19/24).

Aturan yang diuji (pilih bila ...):
  R0 status-quo : hull_LCB <= 0,05
  R1 mean       : hull_mean <= 0,05
  R2 gate-sd    : hull_LCB <= 0,05 DAN (sumber MP ATAU sd <= 0,10)
  R3 bias       : mean NOMAD dikoreksi +0,56 (bias rata-rata 4 label
                  NOMAD), lalu hull_mean_terkoreksi <= 0,05
  R4 mean+gate  : hull_mean <= 0,05 DAN sd <= 0,15
  R5 bio-dulu    : GERBANG BIOLOGI DULU (semua unsur di luar Ca/P/O/H harus
                  ion terverifikasi: Zn, Sr, Mg, Si, Mn, F, Na, K, Cl),
                  LALU R2. Urutan diputar 2026-11-22 berdasar analisis
                  efektivitas zoom 0-4: biologi (presisi 50%) didahulukan
                  atas GNN (presisi 20%) — yang tak lolos biologi tidak
                  pernah sampai ke peringkat GNN.
PERINGATAN JUJUR: backtest hanya n=5 (1 positif) — aturan apa pun yang
"menang" di sini bisa jadi kebetulan. R3 khususnya diestimasi dari 4
label yang sama dengan yang diuji (in-sample). Keputusan adopsi harus
menimbang itu, bukan hanya skor backtest.
Keluaran: aturan_seleksi_v2.json (per kandidat + ringkasan aturan).
"""
import json, os
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = f"{BASE}/data_raw"

def load_hull_entries():
    from pymatgen.core import Composition
    rows = json.load(open(f"{RAW}/mp_hull_subsystems.json"))
    best = {}
    for r in rows:
        comp = Composition(r["formula_pretty"])
        key = comp.reduced_formula
        e = r["formation_energy_per_atom"]
        if key not in best or e < best[key][0]:
            best[key] = (e, comp)
    for el in ("F", "H", "O"):
        if el not in best:
            best[el] = (0.0, Composition(el))
    return best

def build_hull_for(els, hull):
    els = set(els)
    entries = [ComputedEntry(comp, e * comp.num_atoms)
               for e, comp in hull.values()
               if set(str(x) for x in comp.elements).issubset(els)]
    return PhaseDiagram(entries)

man = json.load(open(f"{BASE}/dft_r2/manifest.json"))
jobs = []
def walk(o):
    if isinstance(o, dict):
        if "job" in o and "lcb" in o:
            jobs.append(o); return
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(man)
print(f"Kandidat putaran 2: {len(jobs)}")

ood = {r["job"]: r for r in json.load(open(f"{BASE}/dft_r2/analisis_out_of_domain.json"))["rows"]}
labels = {r["job"]: r for r in json.load(open(f"{BASE}/dft_r2/hasil_analisis_r2.json"))}
print(f"Label DFT: {len(labels)} ({', '.join(sorted(labels))})")

hull = load_hull_entries()
pd_cache = {}
def hull_implied(job, e_form):
    comp = Composition(job["formula"])
    els = set(str(x) for x in comp.elements)
    key = frozenset(els)
    if key not in pd_cache:
        pd_cache[key] = build_hull_for(els, hull)
    entry = ComputedEntry(comp, e_form * comp.num_atoms)
    # allow_negative=True: prediksi GNN (apalagi LCB) bisa jatuh DI BAWAH
    # hull MP — itu justru sinyalnya; tanpa ini pymatgen melempar ValueError.
    return round(pd_cache[key].get_e_above_hull(entry, allow_negative=True), 4)

# Gerbang biologi (daftar ion terverifikasi pemilik, 2026-11-20):
# semua unsur di luar Ca/P/O/H harus termasuk ion terverifikasi.
# C (karbonat) BELUM ada di daftar -> saat ini ditolak; lihat
# analisis-efektivitas-zoom.md (keputusan terbuka).
ION_TERVERIFIKASI = {"Zn", "Sr", "Mg", "Si", "Mn", "F", "Na", "K", "Cl"}
UNSUR_DASAR = {"Ca", "P", "O", "H"}
def lolos_biologi(formula):
    els = {str(x) for x in Composition(formula).elements}
    return (els - UNSUR_DASAR) <= ION_TERVERIFIKASI

rows = []
for j in jobs:
    o = ood.get(j["job"], {})
    r = {"job": j["job"], "formula": j["formula"], "rank": j["rank"], "tier": j["tier"],
         "source": j["source"], "gnn_mean": j["mean_eform"], "gnn_sd": j["sd"],
         "gnn_lcb": j["lcb"], "d_comp": o.get("d_comp"), "f_rare": o.get("f_rare"),
         "bio_lolos": lolos_biologi(j["formula"]),
         "hull_mean": hull_implied(j, j["mean_eform"]),
         "hull_lcb": hull_implied(j, j["lcb"])}
    lab = labels.get(j["job"])
    # hanya berlabel bila e_above_hull-nya ada (r2_18 tertahan mu_Mg -> bukan label)
    if lab and lab.get("e_above_hull") is not None:
        r["dft_hull"] = lab.get("e_above_hull")
        r["stabil_dft"] = bool(lab["e_above_hull"] <= 0.05)
    rows.append(r)

# bias NOMAD dari label (in-sample — laporkan apa adanya)
# bias = rata-rata (DFT_Eform - GNN_mean) = rata-rata (dft_hull - hull_mean)
bias_vals = [r["dft_hull"] - r["hull_mean"] for r in rows
             if r["source"] == "NOMAD" and r.get("dft_hull") is not None]
BIAS_NOMAD = sum(bias_vals) / len(bias_vals) if bias_vals else 0.0
print(f"Bias NOMAD (DFT - GNN mean, n={len(bias_vals)}): +{BIAS_NOMAD:.4f} eV/atom")

def pilih(r, rule):
    if rule == "R0":
        return r["hull_lcb"] <= 0.05
    if rule == "R1":
        return r["hull_mean"] <= 0.05
    if rule == "R2":
        return r["hull_lcb"] <= 0.05 and (r["source"] == "MP" or r["gnn_sd"] <= 0.10)
    if rule == "R3":
        m = r["gnn_mean"] + (BIAS_NOMAD if r["source"] == "NOMAD" else 0.0)
        return hull_implied({"formula": r["formula"]}, m) <= 0.05
    if rule == "R4":
        return r["hull_mean"] <= 0.05 and r["gnn_sd"] <= 0.15
    if rule == "R5":
        # urutan diputar: gerbang biologi dulu, baru R2
        return r["bio_lolos"] and r["hull_lcb"] <= 0.05 and (r["source"] == "MP" or r["gnn_sd"] <= 0.10)
    raise ValueError(rule)

print("\n== BACKTEST di 5 label DFT (TP = stabil yang dipilih, FP = tidak stabil yang dipilih) ==")
summary = {}
for rule in ("R0", "R1", "R2", "R3", "R4", "R5"):
    lab = [r for r in rows if "stabil_dft" in r]
    tp = sum(1 for r in lab if pilih(r, rule) and r["stabil_dft"])
    fp = sum(1 for r in lab if pilih(r, rule) and not r["stabil_dft"])
    fn = sum(1 for r in lab if not pilih(r, rule) and r["stabil_dft"])
    sisa = [r["job"] for r in rows if "stabil_dft" not in r and pilih(r, rule)]
    summary[rule] = {"TP": tp, "FP": fp, "FN": fn, "dipilih_dari_sisa_25": len(sisa), "sisa": sisa}
    print(f"{rule}: TP={tp} FP={fp} FN={fn} | memilih {len(sisa)}/25 sisa: {sisa}")
    for r in lab:
        print(f"   {'PILIH' if pilih(r, rule) else 'tolak':5s} {r['job']} "
              f"(DFT {'STABIL' if r['stabil_dft'] else 'tidak stabil'} {r['dft_hull']:+.4f}; "
              f"hull_mean {r['hull_mean']:+.4f}, hull_LCB {r['hull_lcb']:+.4f}, "
              f"sd {r['gnn_sd']:.3f}, {r['source']})")

print("\n== PERINGKAT SISA (diurutkan hull_mean; bintang = dipilih R2) ==")
sisa = [r for r in rows if "stabil_dft" not in r]
for r in sorted(sisa, key=lambda x: x["hull_mean"])[:12]:
    print(f"  {r['job']} rank {r['rank']} {r['source']} tier {r['tier']} | "
          f"hull_mean {r['hull_mean']:+.4f} hull_LCB {r['hull_lcb']:+.4f} "
          f"sd {r['gnn_sd']:.3f} d_comp {r['d_comp']} {'<-- R2' if pilih(r, 'R2') else ''}")

json.dump({"bias_nomad": BIAS_NOMAD, "aturan": summary, "kandidat": rows},
          open(f"{BASE}/aturan_seleksi_v2.json", "w"), indent=2)
print("\nWrote aturan_seleksi_v2.json")
