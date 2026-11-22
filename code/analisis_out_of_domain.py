#!/usr/bin/env python3
"""Analisis kegagalan di luar domain (out-of-domain) — metrik jarak yang
DI-REGISTRASI SEBELUM label baru datang (rencana-analisis-out-of-domain.md).

Tiga sumbu jarak dari data latih (815 struktur MP Ca-P), dihitung dari
rumus + ukuran sel SAJA (tanpa melihat label DFT):
  d_comp : jarak komposisi L1 ke rumus latih terdekat (vektor komposisi
           ternormalisasi, rentang 0-2; 0 = rumus sudah ada di latih).
  n_rare : jumlah unsur dalam rumus yang muncul di <5% rumus latih
           (<41 dari 815). f_rare: fraksi atom dari unsur-unsur itu.
  nsites : ukuran sel kandidat (dari draf v2).
Hasil (bila sudah ada label DFT):
  |err|  : |E_form DFT - E_pred GNN mean| (eV/atom).
  flagged: sd GNN >= 0,15 eV/atom (ambang pra-registrasi: ~3x rerata sd
           uji 0,0525; ~2x MAE uji 0,087).
Korelasi Spearman hanya dihitung bila n_berlabel >= 5; bila kurang,
skrip mencetak tabel + peringatan 'terlalu sedikit untuk klaim'.

Jalankan: .venv-wp2/bin/python analisis_out_of_domain.py
Keluaran: dft_r2/analisis_out_of_domain.json + tabel tercetak.
"""
import json, os
import numpy as np
from pymatgen.core import Composition

BASE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(BASE, "dft_r2")

RARE_FRAC = 0.05
SD_FLAG = 0.15


def comp_vector(counts, elements):
    tot = sum(counts.values())
    return np.array([counts.get(e, 0.0) / tot for e in elements], float)


def main():
    ds = json.load(open(os.path.join(BASE, "wp2_dataset.json")))
    train_formulas = []
    for e in ds["mp_entries"]:
        train_formulas.append(Composition(e["formula_pretty"]).as_dict())
    elements = sorted({el for c in train_formulas for el in c})
    train_vecs = np.array([comp_vector(c, elements) for c in train_formulas])
    el_freq = {el: sum(1 for c in train_formulas if el in c) / len(train_formulas)
               for el in elements}
    rare_els = {el for el, f in el_freq.items() if f < RARE_FRAC}
    print(f"Latih: {len(train_formulas)} rumus; unsur langka (<5%): {sorted(rare_els)}")

    draft = json.load(open(os.path.join(BASE, "putaran2_draf30_v2.json")))
    labels = {}
    lp = os.path.join(R2, "hasil_analisis_r2.json")
    if os.path.exists(lp):
        for r in json.load(open(lp)):
            labels[r["job"]] = r

    rows = []
    for c in draft:
        counts = Composition(c["formula"]).as_dict()
        vec = comp_vector(counts, elements)
        d_comp = float(np.abs(train_vecs - vec).sum(axis=1).min())
        rare_in = [el for el in counts if el in rare_els]
        n_atoms_rare = sum(counts[el] for el in rare_in)
        f_rare = n_atoms_rare / sum(counts.values())
        job = f"r2_{draft.index(c) + 1:02d}_{c['formula']}"
        row = {"job": job, "formula": c["formula"], "tier": c["tier"],
               "source": c["source"], "nsites": c["nsites"],
               "gnn_mean": c["mean_eform"], "gnn_sd": c["sd"],
               "d_comp": round(d_comp, 3), "n_rare": len(rare_in),
               "rare_elements": rare_in, "f_rare": round(f_rare, 3),
               "flagged_high_sd": c["sd"] >= SD_FLAG}
        lab = labels.get(job)
        if lab and lab.get("e_form_qe_corr") is not None:
            row["E_form_DFT"] = lab["e_form_qe_corr"]
            row["abs_err"] = round(abs(lab["e_form_qe_corr"] - c["mean_eform"]), 4)
            row["e_above_hull"] = lab.get("e_above_hull")
        rows.append(row)

    lab_rows = [r for r in rows if "abs_err" in r]
    print(f"\nKandidat putaran 2: {len(rows)}; berlabel DFT: {len(lab_rows)}")
    print(f"{'job':28s} {'d_comp':>6s} {'n_rare':>6s} {'f_rare':>6s} {'nsites':>6s} "
          f"{'sd':>6s} {'flag':>5s} {'|err|':>7s} {'hull':>7s}")
    for r in rows:
        err_s = f"{r['abs_err']:.4f}" if 'abs_err' in r else '-'
        hull_s = f"{r['e_above_hull']:+.3f}" if r.get('e_above_hull') is not None else '-'
        print(f"{r['job']:28s} {r['d_comp']:6.3f} {r['n_rare']:6d} {r['f_rare']:6.3f} "
              f"{r['nsites']:6d} {r['gnn_sd']:6.3f} "
              f"{str(r['flagged_high_sd']):>5s} {err_s:>7s} {hull_s:>7s}")
    if len(lab_rows) >= 5:
        try:
            from scipy.stats import spearmanr
            for axis in ("d_comp", "n_rare", "f_rare", "nsites", "gnn_sd"):
                rho, p = spearmanr([r[axis] for r in lab_rows],
                                   [r["abs_err"] for r in lab_rows])
                print(f"Spearman |err| vs {axis}: rho={rho:+.3f} p={p:.3f}")
        except Exception as ex:
            print(f"korelasi gagal: {ex}")
    else:
        print(f"\nn_berlabel = {len(lab_rows)} < 5 — TERLALU SEDIKIT untuk korelasi; "
              "klaim ditunda sampai label putaran 2 terkumpul.")
    json.dump({"rare_elements_train": sorted(rare_els), "sd_flag_threshold": SD_FLAG,
               "rows": rows}, open(os.path.join(R2, "analisis_out_of_domain.json"), "w"),
              indent=2)
    print("tersimpan: dft_r2/analisis_out_of_domain.json")


if __name__ == "__main__":
    main()
