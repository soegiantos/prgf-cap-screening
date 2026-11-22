#!/usr/bin/env python3
"""WP1 consolidation — PRGF baru, dataset v1.0 (17 Nov 2026).
Inputs : data_raw/aflow_ca_p.jsonl, mp_ca_p.jsonl, nomad_ca_p.jsonl,
         cod_ca_p.json, pubchem_ca_p.jsonl, oqmd_ca_p.json (EXCLUDED, see below)
Outputs: consolidated_ca_p.jsonl (one row per unique reduced formula),
         consolidated_summary.json, and a printed summary.

OQMD EXCLUSION (verified 2026-11-17): the harvested oqmd_ca_p.json contains
0/1000 records with Ca and P. The oqmdapi formationenergy endpoint ignored
the element_set/elements filters and returned the first 1000 DB entries
(pure elements Lu, Tm, Ne, ...); composition queries return 502/timeout.
OQMD is therefore EXCLUDED from v1.0. It is not silently dropped: the file
is kept, the reason is recorded here and in the report, and re-harvest is a
follow-up task (per-composition queries or the oqmdapi client on the owner's PC).

Energy honesty: AFLOW `energy_atom` is a TOTAL energy per atom, NOT a
formation energy. It is never mixed with MP formation energies. Cross-source
calibration (MP formation E vs AFLOW total E) is done only via a fitted
per-element reference correction (least squares, mechanism C groundwork),
and its residuals are reported.
"""
import json, re, os, math
from collections import Counter, defaultdict
from math import gcd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__)) + "/data_raw"
OUT = os.path.dirname(os.path.abspath(__file__))

EL_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

def parse_formula(s):
    """Return dict element->int count, or (None, reason)."""
    if not s or not isinstance(s, str):
        return None, "empty"
    t = s.strip().strip("-").strip()
    if not t:
        return None, "empty"
    if "+" in t or t.count("-") > 0:
        # charge artefacts, e.g. PubChem 'Ca4O4P+5'
        return None, "charge artefact"
    # parenthesis groups, e.g. Ca3(PO4)2
    while "(" in t:
        m = re.search(r"\(([^()]*)\)(\d*\.?\d*)", t)
        if not m:
            return None, "unparseable parentheses: %s" % s
        inner = parse_formula(m.group(1))
        if inner[0] is None:
            return None, inner[1]
        mult = float(m.group(2)) if m.group(2) else 1.0
        expanded = " ".join(f"{el}{c * mult:g}" for el, c in inner[0].items())
        t = t[:m.start()] + " " + expanded + " " + t[m.end():]
    counts = {}
    compact = t.replace(" ", "").replace(",", "")
    # whole-string match: concatenated 'Ag1Ca2P1' and spaced 'Ca H2 O8 P2' both work
    parts = EL_RE.findall(compact)
    consumed = "".join(f"{el}{n}" for el, n in parts)
    if consumed != compact:
        return None, "unparseable formula"
    for el, num in parts:
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    # integerize
    for scale in (1, 10, 100, 1000):
        if all(abs(c * scale - round(c * scale)) < 1e-6 for c in counts.values()):
            return {el: int(round(c * scale)) for el, c in counts.items()}, None
    return None, "non-integer counts: %s" % s

def reduce_counts(counts):
    g = 0
    for c in counts.values():
        g = gcd(g, c)
    return {el: c // g for el, c in counts.items()} if g > 1 else dict(counts)

def canon(counts):
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))

def hill(counts):
    order = []
    if "C" in counts: order.append("C")
    if "H" in counts: order.append("H")
    order += [el for el in sorted(counts) if el not in ("C", "H")]
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order)

# ---------------- load sources ----------------
recs = []  # (source, source_id, formula_raw, counts|None, parse_note, payload)
def add(source, sid, fraw, payload):
    counts, note = parse_formula(fraw)
    recs.append({"source": source, "id": sid, "formula_raw": fraw,
                 "counts": counts, "note": note, "payload": payload})

for line in open(f"{BASE}/aflow_ca_p.jsonl"):
    d = json.loads(line)
    add("AFLOW", d.get("auid"), d.get("compound"),
        {"energy_atom": d.get("energy_atom"), "enthalpy_atom": d.get("enthalpy_atom"),
         "density": d.get("density"), "sg": d.get("spacegroup_relax"),
         "pearson": d.get("Pearson_symbol_relax"), "species": d.get("species")})
for line in open(f"{BASE}/mp_ca_p.jsonl"):
    d = json.loads(line)
    add("MP", d.get("material_id"), d.get("formula_pretty"),
        {"formation_e": d.get("formation_energy_per_atom"), "e_hull": d.get("energy_above_hull"),
         "density": d.get("density"), "band_gap": d.get("band_gap"),
         "is_stable": d.get("is_stable"), "nsites": d.get("nsites"),
         "symmetry": (d.get("symmetry") or {}).get("symbol"),
         "chemsys": d.get("chemsys")})
for line in open(f"{BASE}/nomad_ca_p.jsonl"):
    d = json.loads(line)
    add("NOMAD", d.get("entry_id"), d.get("formula_reduced"),
        {"formula_hill": d.get("formula_hill"), "elements": d.get("elements")})
cod = json.load(open(f"{BASE}/cod_ca_p.json"))
for d in cod:
    add("COD", str(d.get("file")), d.get("formula"),
        {"sg": d.get("sg"), "sgNumber": d.get("sgNumber"), "vol": d.get("vol"),
         "a": d.get("a"), "b": d.get("b"), "c": d.get("c"),
         "chemname": d.get("chemname"), "doi": d.get("doi"),
         "calcformula": d.get("calcformula"), "duplicateof": d.get("duplicateof")})
for line in open(f"{BASE}/pubchem_ca_p.jsonl"):
    d = json.loads(line)
    add("PubChem", str(d.get("CID")), d.get("MolecularFormula"),
        {"iupac": d.get("IUPACName"), "mw": d.get("MolecularWeight")})

# OQMD: parse only to PROVE the exclusion
oq = json.load(open(f"{BASE}/oqmd_ca_p.json"))["data"]
oq_valid = 0
for d in oq:
    c, _ = parse_formula(d.get("composition"))
    if c and "Ca" in c and "P" in c:
        oq_valid += 1

# ---------------- aggregate by reduced formula ----------------
agg = {}
unparseable = Counter()
per_source = Counter()
per_source_parsed = Counter()
for r in recs:
    per_source[r["source"]] += 1
    if r["counts"] is None:
        unparseable[(r["source"], r["note"])] += 1
        continue
    per_source_parsed[r["source"]] += 1
    red = reduce_counts(r["counts"])
    key = canon(red)
    a = agg.setdefault(key, {"reduced_formula": key, "hill": hill(red),
                             "counts": red, "n_records": 0, "sources": {},
                             "mp": [], "aflow": [], "cod": [], "nomad": [], "pubchem": []})
    a["n_records"] += 1
    a["sources"][r["source"]] = a["sources"].get(r["source"], 0) + 1
    a[r["source"].lower()].append({"id": r["id"], "formula_raw": r["formula_raw"], **r["payload"]})

print(f"Records loaded: {len(recs)}")
print(f"Per source loaded: {dict(per_source)}")
print(f"Per source parsed: {dict(per_source_parsed)}")
print(f"Unparseable (by source, reason): {dict(unparseable)}")
print(f"OQMD records with Ca AND P: {oq_valid} / {len(oq)}  -> EXCLUDED from v1.0")
print(f"Unique reduced formulas: {len(agg)}")

# overlap matrix
sources = ["AFLOW", "MP", "NOMAD", "COD", "PubChem"]
overlap = {}
for s in sources:
    overlap[s] = {t: sum(1 for a in agg.values() if s in a["sources"] and t in a["sources"])
                  for t in sources}
print("Overlap matrix (shared reduced formulas):")
for s in sources:
    print(" ", s, overlap[s])

# ---------------- energy calibration MP vs AFLOW ----------------
# Pair most-stable MP record (lowest e_hull) with most-stable AFLOW record
# (lowest total energy per atom) per shared formula. Fit per-element mu:
# E_form_total(MP) = E_total(AFLOW) - sum n_i mu_i
pairs = []
for key, a in agg.items():
    if "MP" in a["sources"] and "AFLOW" in a["sources"]:
        mp_ok = [x for x in a["mp"] if x.get("formation_e") is not None and x.get("e_hull") is not None]
        af_ok = [x for x in a["aflow"] if x.get("energy_atom") is not None]
        if not mp_ok or not af_ok:
            continue
        mp_best = min(mp_ok, key=lambda x: x["e_hull"])
        af_best = min(af_ok, key=lambda x: x["energy_atom"])
        n_atoms = sum(a["counts"].values())
        pairs.append((key, a["counts"], mp_best["formation_e"] * n_atoms,
                      af_best["energy_atom"] * n_atoms, n_atoms))
print(f"\nMP-AFLOW overlapping formulas with energies: {len(pairs)}")
calib = {}
if len(pairs) >= 20:
    els = sorted({el for _, c, _, _, _ in pairs for el in c})
    A, b = [], []
    for _, c, ef, et, _ in pairs:
        A.append([c.get(el, 0) for el in els])
        b.append(et - ef)
    A, b = np.array(A, float), np.array(b, float)
    mu, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ mu
    resid = b - pred
    n_atoms_tot = np.array([p[4] for p in pairs], float)
    rmse_per_atom = float(np.sqrt(np.mean((resid / n_atoms_tot) ** 2)))
    ss_tot = float(np.sum((b - b.mean()) ** 2)); ss_res = float(np.sum(resid ** 2))
    calib = {"n_overlap_formulas": len(pairs), "elements_fitted": els,
             "mu_eV_per_atom_of_element": {el: float(m) for el, m in zip(els, mu)},
             "residual_RMSE_eV_per_atom": rmse_per_atom,
             "R2": 1 - ss_res / ss_tot if ss_tot else None,
             "note": ("Fit: E_form(MP,total) = E_total(AFLOW,total) - sum n_i mu_i. "
                      "Pairs match most-stable polymorph per source; residuals include "
                      "polymorph mismatch + code/pseudopotential differences. "
                      "mu values are NOT transferable physical constants without validation.")}
    print("Calibration elements:", els)
    print("mu:", {el: round(float(m), 3) for el, m in zip(els, mu)})
    print(f"Residual RMSE: {rmse_per_atom:.4f} eV/atom, R2: {calib['R2']:.3f}")

# ---------------- data-gap map ----------------
def has_mp_e(a): return any(x.get("formation_e") is not None for x in a["mp"])
def has_density(a): return (any(x.get("density") is not None for x in a["mp"]) or
                            any(x.get("density") is not None for x in a["aflow"]))
def has_structure(a): return ("COD" in a["sources"] or "MP" in a["sources"] or "AFLOW" in a["sources"])
def has_bandgap(a): return any(x.get("band_gap") is not None for x in a["mp"])
gap = {
    "formulas_with_MP_formation_energy": sum(1 for a in agg.values() if has_mp_e(a)),
    "formulas_with_density": sum(1 for a in agg.values() if has_density(a)),
    "formulas_with_band_gap": sum(1 for a in agg.values() if has_bandgap(a)),
    "formulas_with_any_structure": sum(1 for a in agg.values() if has_structure(a)),
    "formulas_with_energy_and_structure": sum(1 for a in agg.values() if has_mp_e(a) and has_structure(a)),
    "formulas_NOMAD_only": sum(1 for a in agg.values() if set(a["sources"]) == {"NOMAD"}),
    "formulas_COD_only": sum(1 for a in agg.values() if set(a["sources"]) == {"COD"}),
    "formulas_AFLOW_only": sum(1 for a in agg.values() if set(a["sources"]) == {"AFLOW"}),
    "formulas_in_2plus_sources": sum(1 for a in agg.values() if len(a["sources"]) >= 2),
    "formulas_in_3plus_sources": sum(1 for a in agg.values() if len(a["sources"]) >= 3),
    "formulas_with_elastic_data": 0,
}
# phosphate / Ca-P chemistry
def ca_p_ratio(a):
    c = a["counts"]
    if "Ca" in c and "P" in c and c["P"]:
        return c["Ca"] / c["P"]
    return None
ratios = [(a["reduced_formula"], ca_p_ratio(a)) for a in agg.values()]
ratios = [(f, r) for f, r in ratios if r is not None]
phos = [a for a in agg.values() if "P" in a["counts"] and "O" in a["counts"]]
chem = {
    "formulas_with_Ca_and_P": len(ratios),
    "formulas_phosphate_P_and_O": len(phos),
    "formulas_Ca_P_ratio_1.5_to_1.67": sum(1 for _, r in ratios if 1.5 <= r <= 1.67),
    "formulas_Ca_P_ratio_near_1.67_pm0.05": sum(1 for _, r in ratios if abs(r - 1.67) <= 0.05),
    "formulas_with_elements_beyond_Ca_P_O_H_F_C": sum(
        1 for a in agg.values() if set(a["counts"]) - {"Ca", "P", "O", "H", "F", "C"}),
    "top_extra_elements": Counter(
        el for a in agg.values() for el in a["counts"] if el not in {"Ca", "P", "O", "H", "F", "C"}).most_common(10),
    "formulas_with_Sr": sum(1 for a in agg.values() if "Sr" in a["counts"]),
    "formulas_with_Mg": sum(1 for a in agg.values() if "Mg" in a["counts"]),
    "formulas_with_Zn": sum(1 for a in agg.values() if "Zn" in a["counts"]),
    "formulas_with_Si": sum(1 for a in agg.values() if "Si" in a["counts"]),
}
print("\nGap map:", json.dumps(gap, indent=1))
print("Chemistry:", json.dumps(chem, indent=1, default=str))

# ---------------- write outputs ----------------
rows = []
for key in sorted(agg):
    a = agg[key]
    rows.append({
        "reduced_formula": a["reduced_formula"], "hill_formula": a["hill"],
        "counts": a["counts"], "n_records": a["n_records"], "sources": a["sources"],
        "mp_material_ids": [x["id"] for x in a["mp"]],
        "mp_formation_energy_per_atom": [x["formation_e"] for x in a["mp"] if x.get("formation_e") is not None],
        "mp_energy_above_hull": [x["e_hull"] for x in a["mp"] if x.get("e_hull") is not None],
        "mp_density": [x["density"] for x in a["mp"] if x.get("density") is not None],
        "mp_band_gap": [x["band_gap"] for x in a["mp"] if x.get("band_gap") is not None],
        "aflow_auids": [x["id"] for x in a["aflow"]],
        "aflow_total_energy_per_atom": [x["energy_atom"] for x in a["aflow"] if x.get("energy_atom") is not None],
        "aflow_density": [x["density"] for x in a["aflow"] if x.get("density") is not None],
        "cod_file_ids": [x["id"] for x in a["cod"]],
        "cod_spacegroups": [x["sg"] for x in a["cod"] if x.get("sg")],
        "nomad_entry_ids": [x["id"] for x in a["nomad"]],
        "pubchem_cids": [x["id"] for x in a["pubchem"]],
        "pubchem_names": [x["iupac"] for x in a["pubchem"] if x.get("iupac")],
    })
with open(f"{OUT}/consolidated_ca_p.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

summary = {
    "dataset": "PRGF baru consolidated v1.0",
    "date": "2026-11-17",
    "records_loaded": len(recs),
    "records_loaded_per_source": dict(per_source),
    "records_parsed_per_source": dict(per_source_parsed),
    "unparseable": {f"{s}: {n}": c for (s, n), c in unparseable.items()},
    "unique_reduced_formulas": len(agg),
    "overlap_matrix_shared_reduced_formulas": overlap,
    "oqmd": {"records_harvested": len(oq), "records_with_Ca_and_P": oq_valid,
             "status": "EXCLUDED from v1.0 — element filters ignored by oqmdapi; composition queries 502/timeout",
             "follow_up": "per-composition re-harvest or oqmdapi client on owner's PC"},
    "calibration_MP_vs_AFLOW": calib,
    "gap_map": gap,
    "chemistry": chem,
    "notes": [
        "AFLOW energy_atom is TOTAL energy per atom, never mixed with MP formation energies.",
        "Same reduced formula from different sources may be different polymorphs; composition-level dedup does not imply structure identity.",
        "COD formula vs calcformula can differ in H count (H placement); COD 'formula' was used, H counts should be treated with care.",
        "NOMAD-only formulas have elements + formula only (no energy, no structure) — they are pointers for follow-up fetch, not training data.",
        "Elastic data: 0 formulas in v1.0 (same gap as old pipeline, n=9 there).",
    ],
}
json.dump(summary, open(f"{OUT}/consolidated_summary.json", "w"), indent=2)
print("\nWrote consolidated_ca_p.jsonl:", len(rows), "rows; consolidated_summary.json")
