#!/usr/bin/env python3
"""WP2 dataset builder — PRGF baru (system python: numpy only).
Inputs : data_raw/mp_structures.json (815 MP structures + formation energies),
         data_raw/cod_cif/*.cif (940 COD structures, no energies),
         consolidated_ca_p.jsonl (reduced formulas)
Outputs: wp2_dataset.json with:
  - mp_entries: {material_id, reduced_formula, lattice, elements, frac_coords,
                 cart_coords, formation_e, e_hull, density, band_gap, nsites}
  - cod_entries: {file_id, reduced_formula, lattice, elements, frac_coords,
                  cart_coords, spacegroup, formula_raw}  (no target)
  - splits: grouped (by reduced formula) train/val/test + 5 GroupKFold folds
CIF parsing is a small purpose-built parser (no pymatgen needed here):
cell lengths/angles + the _atom_site_ loop. Entries that fail parsing are
listed, not silently dropped.
"""
import json, re, os, math, random
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = BASE + "/data_raw"

# ---------- formula helpers (same rules as consolidate.py) ----------
EL_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
def parse_formula(s):
    if not s or not isinstance(s, str):
        return None
    t = s.strip().strip("-").strip()
    if not t or "+" in t or t.count("-") > 0:
        return None
    while "(" in t:
        m = re.search(r"\(([^()]*)\)(\d*\.?\d*)", t)
        if not m:
            return None
        inner = parse_formula(m.group(1))
        if inner is None:
            return None
        mult = float(m.group(2)) if m.group(2) else 1.0
        t = t[:m.start()] + " " + " ".join(f"{el}{c*mult:g}" for el, c in inner.items()) + " " + t[m.end():]
    compact = t.replace(" ", "").replace(",", "")
    parts = EL_RE.findall(compact)
    if "".join(f"{el}{n}" for el, n in parts) != compact:
        return None
    counts = {}
    for el, num in parts:
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    for scale in (1, 10, 100, 1000):
        if all(abs(c*scale - round(c*scale)) < 1e-6 for c in counts.values()):
            return {el: int(round(c*scale)) for el, c in counts.items()}
    return None

def reduce_counts(counts):
    from math import gcd
    g = 0
    for c in counts.values():
        g = gcd(g, c)
    return {el: c // g for el, c in counts.items()} if g > 1 else dict(counts)

def canon(counts):
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))

# ---------- lattice ----------
def lattice_matrix(a, b, c, alpha, beta, gamma):
    al, be, ga = map(math.radians, (alpha, beta, gamma))
    va = [a, 0.0, 0.0]
    vb = [b*math.cos(ga), b*math.sin(ga), 0.0]
    cx = c*math.cos(be)
    cy = c*(math.cos(al) - math.cos(be)*math.cos(ga))/math.sin(ga)
    cz = math.sqrt(max(c*c - cx*cx - cy*cy, 0.0))
    return [va, vb, [cx, cy, cz]]

# ---------- minimal CIF parser ----------
def parse_cif(path):
    text = open(path, errors="ignore").read()
    def single(tag):
        m = re.search(r"^" + re.escape(tag) + r"\s+(\S+)", text, re.M)
        if not m:
            return None
        v = m.group(1).strip("'\"")
        try:
            return float(re.sub(r"\(.*\)$", "", v))
        except Exception:
            return None
    a, b, c = single("_cell_length_a"), single("_cell_length_b"), single("_cell_length_c")
    al, be, ga = single("_cell_angle_alpha"), single("_cell_angle_beta"), single("_cell_angle_gamma")
    sg = None
    m = re.search(r"^_symmetry_space_group_name_H-M\s+'?([^'\n]+)'?", text, re.M)
    if m:
        sg = m.group(1).strip()
    if None in (a, b, c, al, be, ga):
        return None, "missing cell parameters"
    # atom site loop: find loop_ containing _atom_site_fract_x
    lines = text.splitlines()
    sites = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if "_atom_site_fract_x" in cols and "_atom_site_fract_y" in cols and "_atom_site_fract_z" in cols:
                ix, iy, iz = cols.index("_atom_site_fract_x"), cols.index("_atom_site_fract_y"), cols.index("_atom_site_fract_z")
                ilab = cols.index("_atom_site_label") if "_atom_site_label" in cols else None
                ityp = cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in cols else None
                k = j
                while k < len(lines):
                    s = lines[k].strip()
                    if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("data_") or s.startswith("_"):
                        break
                    if s == ";":
                        break
                    toks = s.split()
                    if len(toks) < len(cols):
                        break
                    def _f(tok):
                        # strip CIF uncertainty suffix, e.g. 0.2331(3) -> 0.2331
                        return float(re.sub(r"\(.*\)$", "", tok))
                    try:
                        fx, fy, fz = _f(toks[ix]), _f(toks[iy]), _f(toks[iz])
                    except Exception:
                        # placeholder row such as '? ? ? ?' — skip the row, not the file
                        k += 1
                        continue
                    lab = toks[ilab] if ilab is not None else (toks[ityp] if ityp is not None else "?")
                    el = re.match(r"[A-Z][a-z]?", lab)
                    sites.append({"element": el.group(0) if el else lab, "frac": [fx, fy, fz]})
                    k += 1
                i = k
                continue
            i = j
        else:
            i += 1
    if not sites:
        return None, "no atom sites parsed"
    lat = lattice_matrix(a, b, c, al, be, ga)
    return {"lattice": lat, "sites": sites, "spacegroup": sg}, None

# ---------- MP entries ----------
mp = json.load(open(f"{RAW}/mp_structures.json"))
mp_entries, mp_fail = [], []
for d in mp:
    counts = parse_formula(d.get("formula_pretty"))
    if counts is None:
        mp_fail.append(d.get("material_id")); continue
    st = d.get("structure") or {}
    lat = (st.get("lattice") or {}).get("matrix")
    sites = st.get("sites", [])
    els, frac, cart = [], [], []
    for s in sites:
        sp = s.get("species", [{}])[0]
        el = sp.get("element") or s.get("label", "?")
        els.append(el)
        frac.append(s.get("abc", [0, 0, 0]))
        cart.append(s.get("xyz", [0, 0, 0]))
    mp_entries.append({
        "material_id": d.get("material_id"), "formula_pretty": d.get("formula_pretty"),
        "reduced_formula": canon(reduce_counts(counts)), "counts": counts,
        "lattice": lat, "elements": els, "frac_coords": frac, "cart_coords": cart,
        "formation_e": d.get("formation_energy_per_atom"), "e_hull": d.get("energy_above_hull"),
        "density": d.get("density"), "band_gap": d.get("band_gap"), "nsites": len(els)})
print(f"MP entries: {len(mp_entries)}, failed formula parse: {mp_fail}")

# ---------- COD entries ----------
cod_meta = {str(d["file"]): d for d in json.load(open(f"{RAW}/cod_ca_p.json"))}
cod_entries, cod_fail = [], []
import glob
for path in sorted(glob.glob(f"{RAW}/cod_cif/*.cif")):
    fid = os.path.basename(path)[:-4]
    parsed, err = parse_cif(path)
    meta = cod_meta.get(fid, {})
    if parsed is None:
        cod_fail.append({"file": fid, "reason": err}); continue
    els = [s["element"] for s in parsed["sites"]]
    counts = {}
    for el in els:
        counts[el] = counts.get(el, 0) + 1
    lat = np.array(parsed["lattice"])
    frac = np.array([s["frac"] for s in parsed["sites"]], float)
    cart = (frac @ lat).tolist()
    cod_entries.append({
        "file_id": fid, "formula_raw": meta.get("formula"),
        "reduced_formula": canon(reduce_counts(counts)), "counts": counts,
        "lattice": parsed["lattice"], "elements": els,
        "frac_coords": frac.tolist(), "cart_coords": cart,
        "spacegroup": parsed["spacegroup"], "nsites": len(els)})
print(f"COD entries parsed: {len(cod_entries)}, failed: {len(cod_fail)} -> {cod_fail[:5]}")

# ---------- grouped splits (by reduced formula, leakage-resistant) ----------
groups = sorted({e["reduced_formula"] for e in mp_entries})
rng = random.Random(42)
rng.shuffle(groups)
n = len(groups)
n_test, n_val = int(0.15*n), int(0.15*n)
test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test+n_val])
train_g = set(groups[n_test+n_val:])
def split_of(e):
    return "test" if e["reduced_formula"] in test_g else ("val" if e["reduced_formula"] in val_g else "train")
for e in mp_entries:
    e["split"] = split_of(e)
# 5 GroupKFold-style folds (grouped, seed 42)
folds = []
shuffled = groups[:]
random.Random(42).shuffle(shuffled)
for k in range(5):
    folds.append(sorted(shuffled[k::5]))
from collections import Counter
print("MP split counts (entries):", Counter(e["split"] for e in mp_entries))
print("MP split counts (groups):", {"train": len(train_g), "val": len(val_g), "test": len(test_g)})
print("Target formation_e: n=", sum(1 for e in mp_entries if e["formation_e"] is not None),
      "mean=%.3f sd=%.3f" % (float(np.mean([e["formation_e"] for e in mp_entries if e["formation_e"] is not None])),
                              float(np.std([e["formation_e"] for e in mp_entries if e["formation_e"] is not None]))))

out = {"mp_entries": mp_entries, "cod_entries": cod_entries,
       "cod_failed": cod_fail, "mp_failed_formula": mp_fail,
       "folds_groups": folds,
       "meta": {"n_mp": len(mp_entries), "n_cod": len(cod_entries),
                "n_groups": n, "seed": 42, "split": "70/15/15 grouped by reduced formula"}}
json.dump(out, open(f"{BASE}/wp2_dataset.json", "w"))
print("Wrote wp2_dataset.json: MP", len(mp_entries), "COD", len(cod_entries))
