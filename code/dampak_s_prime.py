#!/usr/bin/env python3
"""Dampak s' pada LCB + aturan seleksi — PRGF baru (bagian 2 kalibrasi s).

Membaca aturan_seleksi_v2.json (30 kandidat: gnn_mean, gnn_sd,
hull_mean, hull_lcb dgn s=1,85, bio_lolos, source, label DFT).
Menghitung ulang hull_LCB dengan s' via pymatgen (jalur yang sama
dengan aturan_seleksi_v2.py) dan membandingkan:
  - pool R2 (hull_LCB <= 0,05 dan (MP atau sd <= 0,10))
  - backtest R0/R2/R5 pada 5 label
  - seleksi prospektif R5 di antara 25 tak berlabel
  - top-3 R6/R6b (beberapa varian kolam/normalisasi) vs yang dibekukan
    (R5: r2_18, r2_25, r2_27; R6: r2_26, r2_29, r2_18;
     R6b: r2_18, r2_25, r2_27)
Juga memverifikasi identitas linear hull_lcb = hull_mean - s*sd
(get_e_above_hull linear terhadap energi pembentukan).
"""
import json, os
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = f"{BASE}/data_raw"
S_LAMA = 1.85


def load_hull_entries():
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


def main():
    kal = json.load(open(f"{BASE}/kalibrasi_s_validasi.json"))
    s_prime = kal["s_prime"]
    d = json.load(open(f"{BASE}/aturan_seleksi_v2.json"))
    rows = d["kandidat"]
    assert len(rows) == 30
    hull = load_hull_entries()
    pd_cache = {}

    def hull_implied(formula, e_form):
        comp = Composition(formula)
        els = set(str(x) for x in comp.elements)
        key = frozenset(els)
        if key not in pd_cache:
            pd_cache[key] = build_hull_for(els, hull)
        entry = ComputedEntry(comp, e_form * comp.num_atoms)
        return round(pd_cache[key].get_e_above_hull(entry, allow_negative=True), 4)

    # --- hitung ulang hull_lcb dengan s' (pymatgen) + cek identitas linear ---
    max_dev = 0.0
    for r in rows:
        lcb_baru = r["gnn_mean"] - s_prime * r["gnn_sd"]
        hl_pym = hull_implied(r["formula"], lcb_baru)
        hl_lin = round(r["hull_mean"] - s_prime * r["gnn_sd"], 4)
        max_dev = max(max_dev, abs(hl_pym - hl_lin))
        r["hull_lcb_baru"] = hl_pym
        r["hull_lcb_linear"] = hl_lin
        # verifikasi ulang s lama mereproduksi hull_lcb tersimpan
        hl_lama = hull_implied(r["formula"], r["gnn_mean"] - S_LAMA * r["gnn_sd"])
        assert abs(hl_lama - r["hull_lcb"]) <= 0.0002, (r["job"], hl_lama, r["hull_lcb"])
    print(f"Verifikasi s=1,85 mereproduksi hull_lcb tersimpan: OK (30/30)")
    print(f"Identitas linear hull_lcb = hull_mean - s*sd: max |selisih| vs pymatgen = {max_dev:.4f}")

    lab = [r for r in rows if "stabil_dft" in r]
    unl = [r for r in rows if "stabil_dft" not in r]
    assert len(lab) == 5 and len(unl) == 25

    def pilih(r, rule, s_key):
        hl = r["hull_lcb"] if s_key == "lama" else r["hull_lcb_baru"]
        if rule == "R0":
            return hl <= 0.05
        if rule == "R1":
            return r["hull_mean"] <= 0.05
        if rule == "R2":
            return hl <= 0.05 and (r["source"] == "MP" or r["gnn_sd"] <= 0.10)
        if rule == "R5":
            return r["bio_lolos"] and hl <= 0.05 and (r["source"] == "MP" or r["gnn_sd"] <= 0.10)
        raise ValueError(rule)

    def backtest(rule, s_key):
        tp = sum(1 for r in lab if pilih(r, rule, s_key) and r["stabil_dft"])
        fp = sum(1 for r in lab if pilih(r, rule, s_key) and not r["stabil_dft"])
        fn = sum(1 for r in lab if not pilih(r, rule, s_key) and r["stabil_dft"])
        return {"TP": tp, "FP": fp, "FN": fn}

    print("\n== BACKTEST 5 label: s=1,85 vs s'=%.2f ==" % s_prime)
    bt = {}
    for rule in ("R0", "R1", "R2", "R5"):
        b_lama = backtest(rule, "lama")
        b_baru = backtest(rule, "baru")
        bt[rule] = {"s185": b_lama, "s_prime": b_baru,
                    "berubah": b_lama != b_baru}
        print(f"  {rule}: lama TP={b_lama['TP']} FP={b_lama['FP']} FN={b_lama['FN']} | "
              f"baru TP={b_baru['TP']} FP={b_baru['FP']} FN={b_baru['FN']} | "
              f"{'BERUBAH' if b_lama != b_baru else 'tidak berubah'}")

    print("\n== POOL R2 (hull_LCB<=0,05 dan (MP atau sd<=0,10)) ==")
    pool_lama = [r["job"] for r in rows if pilih(r, "R2", "lama")]
    pool_baru = [r["job"] for r in rows if pilih(r, "R2", "baru")]
    print(f"  lama ({len(pool_lama)}): {pool_lama}")
    print(f"  baru ({len(pool_baru)}): {pool_baru}")
    print(f"  masuk: {sorted(set(pool_baru) - set(pool_lama))}; keluar: {sorted(set(pool_lama) - set(pool_baru))}")

    print("\n== SELEKSI R5 di antara 25 tak berlabel ==")
    r5_lama = [r["job"] for r in unl if pilih(r, "R5", "lama")]
    r5_baru = [r["job"] for r in unl if pilih(r, "R5", "baru")]
    print(f"  lama ({len(r5_lama)}): {r5_lama}")
    print(f"  baru ({len(r5_baru)}): {r5_baru}")
    print(f"  masuk: {sorted(set(r5_baru) - set(r5_lama))}; keluar: {sorted(set(r5_lama) - set(r5_baru))}")

    # --- R6 / R6b: skor dari hitung_r6.py (bio + faktor), LCB baru ---
    SKOR_ION = {"Zn": 2, "Sr": 2, "Mg": 2, "Si": 2, "Mn": 1, "F": 1, "Na": 1, "K": 1, "Cl": 1}
    DASAR = {"Ca", "P", "O", "H"}

    def bio_score(formula):
        els = {str(x) for x in Composition(formula).elements}
        asing = els - DASAR
        return 2 if not asing else min(SKOR_ION.get(e, 0) for e in asing)

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

    def skor_kolam(pool, norm_pool, s_key):
        hls = [(r["hull_lcb"] if s_key == "lama" else r["hull_lcb_baru"]) for r in norm_pool]
        lo, hi = min(hls), max(hls)
        out = []
        for r in pool:
            hl = r["hull_lcb"] if s_key == "lama" else r["hull_lcb_baru"]
            s_stab = (hi - hl) / (hi - lo)
            out.append({**r, "s_stab": round(s_stab, 3),
                        "S6": round(0.80 * s_stab + 0.20 * (r["bio"] / 2), 3),
                        "S6b": round(s_stab * r["faktor"], 3)})
        return out

    def top3(pool_skor, kunci):
        return [r["job"] for r in sorted(pool_skor, key=lambda x: (-x[kunci], x["job"]))[:3]]

    pool9_lama = [r for r in rows if r["job"] in pool_lama and "stabil_dft" not in r]
    pool9_baru = [r for r in rows if r["job"] in pool_baru and "stabil_dft" not in r]
    frozen = {"R6": ["r2_26_BCaO5P", "r2_29_Ca9InO28P7", "r2_18_CaFMgO4P"],
              "R6b": ["r2_18_CaFMgO4P", "r2_25_Ca5F4Mg5O21P5", "r2_27_Ca4F3Mg4O17P4"]}

    print("\n== TOP-3 R6 / R6b (varian kolam & normalisasi) ==")
    varian = {}
    # (a) primer: kolam R2 baru, norm atas kolam itu (dengan s')
    for nama, pool, norm, s_key in [
        ("primer_kolam_baru_norm_baru", pool9_baru, pool9_baru, "baru"),
        ("kolam9_lama_norm9_lama_s_baru", pool9_lama, pool9_lama, "baru"),
        ("sensA_kolam9_norm30_s_baru", pool9_lama, rows, "baru"),
        ("sensB_kolam25_norm25_s_baru", unl, unl, "baru"),
        ("kontrol_primer_s_lama", pool9_lama, pool9_lama, "lama"),
    ]:
        sk = skor_kolam(pool, norm, s_key)
        t6 = top3(sk, "S6")
        t6b = top3(sk, "S6b")
        varian[nama] = {"R6": t6, "R6b": t6b,
                        "R6_sama_dengan_beku": t6 == frozen["R6"],
                        "R6b_sama_dengan_beku": t6b == frozen["R6b"]}
        pad = " " * (len(nama) + 2)
        print(f"  {nama}: R6={t6} {'(sama dgn beku)' if t6 == frozen['R6'] else '(BERUBAH vs beku)'}")
        print(f"  {pad}R6b={t6b} {'(sama dgn beku)' if t6b == frozen['R6b'] else '(BERUBAH vs beku)'}")

    # detail: kandidat yang hull_lcb-nya melewati ambang 0,05
    print("\n== Kandidat yang melewati ambang hull_LCB 0,05 (lama -> baru) ==")
    for r in rows:
        if (r["hull_lcb"] <= 0.05) != (r["hull_lcb_baru"] <= 0.05):
            print(f"  {r['job']}: hull_lcb {r['hull_lcb']:+.4f} -> {r['hull_lcb_baru']:+.4f} "
                  f"(hull_mean {r['hull_mean']:+.4f}, sd {r['gnn_sd']:.4f}, {r['source']}, "
                  f"bio {'lolos' if r['bio_lolos'] else 'tolak'}, "
                  f"{'label: ' + ('STABIL' if r['stabil_dft'] else 'tidak stabil') if 'stabil_dft' in r else 'tak berlabel'})")

    dampak = {
        "s_prime": s_prime, "s_lama": S_LAMA,
        "verifikasi_s_lama_mereproduksi_hull_lcb": "30/30 OK",
        "identitas_linear_max_selisih": round(max_dev, 4),
        "backtest": bt,
        "pool_R2_lama": pool_lama, "pool_R2_baru": pool_baru,
        "pool_R2_masuk": sorted(set(pool_baru) - set(pool_lama)),
        "pool_R2_keluar": sorted(set(pool_lama) - set(pool_baru)),
        "R5_tak_berlabel_lama": r5_lama, "R5_tak_berlabel_baru": r5_baru,
        "R5_masuk": sorted(set(r5_baru) - set(r5_lama)),
        "R5_keluar": sorted(set(r5_lama) - set(r5_baru)),
        "R6_R6b_varian": varian,
        "beku_R6": frozen["R6"], "beku_R6b": frozen["R6b"],
        "hull_lcb_baru_per_kandidat": {r["job"]: r["hull_lcb_baru"] for r in rows},
    }
    kal["dampak_lcb"] = dampak
    json.dump(kal, open(f"{BASE}/kalibrasi_s_validasi.json", "w"), indent=2)
    print("\nWrote kalibrasi_s_validasi.json (lengkap: kalibrasi + dampak LCB)")


if __name__ == "__main__":
    main()
