#!/usr/bin/env python3
"""WP2 baseline komposisi — PRGF baru (venv python: sklearn + pymatgen).
Target: formation_energy_per_atom (MP, 815 entries).
Fitur: fraksi unsur + rata-rata tertimbang properti unsur (Magpie-lite:
Z, elektronegativitas Pauling, jari-jari atom, massa, energi ionisasi)
+ nsites + densitas. Bukan fitur struktur — ini pembanding yang harus
dikalahkan GNN.
Evaluasi: 5-fold GroupKFold (grup = rumus tereduksi, dari wp2_dataset.json)
+ test set terpisah (115 entri, 106 rumus tak terlihat saat latih).
Model: SVR (RBF) dan RandomForest.
"""
import json, numpy as np
from pymatgen.core import Element

BASE = "/home/hatch/workspace/research/PRGF baru"
ds = json.load(open(f"{BASE}/wp2_dataset.json"))
mp = ds["mp_entries"]
folds = ds["folds_groups"]

ELEMENTS = sorted({el for e in mp for el in e["counts"]})
ELIDX = {el: i for i, el in enumerate(ELEMENTS)}
PROP = ["Z", "X", "atomic_radius", "atomic_mass", "ionization_energy"]
_cache = {}
def elprops(el):
    if el not in _cache:
        E = Element(el)
        _cache[el] = {
            "Z": E.Z,
            "X": E.X if E.X is not None else 0.0,
            "atomic_radius": E.atomic_radius if E.atomic_radius is not None else 0.0,
            "atomic_mass": E.atomic_mass,
            "ionization_energy": E.ionization_energy if E.ionization_energy is not None else 0.0,
        }
    return _cache[el]

def featurize(e):
    counts = e["counts"]; total = sum(counts.values())
    frac = np.zeros(len(ELEMENTS))
    for el, c in counts.items():
        frac[ELIDX[el]] = c / total
    stats = []
    for p in PROP:
        vals = np.array([elprops(el)[p] for el in counts])
        w = np.array([counts[el] / total for el in counts])
        stats += [float((vals * w).sum()), float(np.sqrt(((vals - (vals * w).sum()) ** 2 * w).sum()))]
    return np.concatenate([frac, stats, [e["nsites"], e["density"] or 0.0]])

X = np.array([featurize(e) for e in mp])
import os
if os.environ.get("WP2_ABLATION") == "comp_only":
    X = X[:, :-2]  # buang nsites + densitas (fitur turunan struktur) -> komposisi murni
    print("ABLASI: komposisi murni (tanpa nsites/densitas)")
y = np.array([e["formation_e"] for e in mp])
groups = np.array([e["reduced_formula"] for e in mp])
splits = np.array([e["split"] for e in mp])
print(f"Fitur: {X.shape[1]} dimensi ({len(ELEMENTS)} unsur + 10 statistik + nsites + densitas)")
print(f"Data: {len(mp)} entri; unsur: {ELEMENTS}")

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def report(name, ytrue, ypred):
    print(f"  {name}: MAE={mean_absolute_error(ytrue, ypred):.4f}  RMSE={mean_squared_error(ytrue, ypred) ** 0.5:.4f}  R2={r2_score(ytrue, ypred):.3f}")
    return {"mae": float(mean_absolute_error(ytrue, ypred)),
            "rmse": float(mean_squared_error(ytrue, ypred) ** 0.5),
            "r2": float(r2_score(ytrue, ypred))}

results = {}
for mname, model in [("SVR-RBF", make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05))),
                     ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=4))]:
    print(f"\n== {mname} — 5-fold GroupKFold (grup=rumus) ==")
    oof = np.zeros_like(y, dtype=float)
    for k, test_g in enumerate(folds):
        te = np.isin(groups, test_g)
        tr = ~te
        m = model
        from sklearn.base import clone
        m = clone(model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    cv = report("CV (out-of-fold)", y, oof)
    print(f"== {mname} — held-out test (115 entri, rumus tak terlihat) ==")
    tr = splits != "test"; te = splits == "test"
    m = clone(model)
    m.fit(X[tr], y[tr])
    tst = report("TEST", y[te], m.predict(X[te]))
    results[mname] = {"cv": cv, "test": tst}

json.dump(results, open(f"{BASE}/wp2_baseline.json", "w"), indent=2)
print("\nWrote wp2_baseline.json")
