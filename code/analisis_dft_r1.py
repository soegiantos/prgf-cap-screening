#!/usr/bin/env python3
"""Analisis DFT WP3 putaran 1 — dari energi total QE ke kesimpulan.

Alur ilmiah (jujur soal keterbatasannya):
  1. --fetch   : ambil energi total MP (energy_per_atom) untuk 815 entri Ca-P
                 + entri fase-pesaing untuk 6 sistem kimia 7 kandidat bersih.
  2. --calibrate: pasang energi referensi unsur mu_i dari data MP (least squares)
                 lalu hitung energi formasi beta-TCP dari energi total QE
                 (ref.out, -689.45963779 Ry) dan bandingkan dengan MP (-3.32566 eV/atom).
                 PERINGATAN YANG AKAN TERBUKTI DI SINI: energi total absolut QE
                 dan VASP TIDAK sebanding (pseudopotential berbeda) — selisih
                 kalibrasi ini mengukur offset antar-kode dan membuktikan bahwa
                 energi formasi QE HARUS memakai referensi unsur QE sendiri
                 (lihat prepare_qe_ref.py), bukan mu dari MP.
  3. --analyze DIR: DIR berisi pasangan <job>.out + <job>.in putaran 1
                 (salin dulu dari mesin DFT). Untuk tiap job yang JOB DONE:
                 energi formasi QE dari mu QE + koreksi anion MP2020
                 (O -0,687; F -0,462; Cl -0,614; I -0,379 eV/atom anion —
                 WAJIB sebelum banding hull, lihat catatan di do_analyze)
                 -> jarak hull (diagram fase MP, 1.908 entri subsistem)
                 -> dft_r1/hasil_analisis_r1.json. Uji validasi: beta-TCP
                 harus memberi E_form(corr) ~-3,27 dan e_above_hull ~0,06.

Pemakaian:
  .venv-wp2/bin/python analisis_dft_r1.py --fetch
  .venv-wp2/bin/python analisis_dft_r1.py --calibrate
  .venv-wp2/bin/python analisis_dft_r1.py --analyze DIR
  .venv-wp2/bin/python analisis_dft_r1.py --mu-qe DIR
Kunci MP dibaca dari ~/.secrets/mp_api_key (tidak pernah dicetak/disimpan di output).
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

RY_TO_EV = 13.605693122994
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data_raw")
KEY_PATH = os.path.expanduser("~/.secrets/mp_api_key")

# 7 kandidat bersih putaran 1 (dari wp3_round1_candidates.json)
CANDIDATES = ["r1_10_CaFK2O4P", "r1_07_CaO3P", "r1_11_C2CaLiO6P",
              "r1_01_BaCa6O17P4", "r1_05_CCa3H4O14P2Zn2", "r1_13_C2CaO9P",
              "r1_02_Ca11O28P7"]
CHEMSYS = ["Ba-Ca-O-P", "Ca-O-P", "C-Ca-H-O-P-Zn", "Ca-F-K-O-P",
           "C-Ca-Li-O-P", "C-Ca-O-P"]

# ---------------- MP fetch ----------------
def mp_get(params, limit=1000):
    import requests, time
    key = open(KEY_PATH).read().strip()
    headers = {"X-API-KEY": key, "User-Agent": "Muse-PRGF-research/1.0 (academic, Universitas Airlangga)"}
    rows, skip = [], 0
    while True:
        p = dict(params); p.update({"_limit": limit, "_skip": skip})
        ok = False
        for att in range(3):
            try:
                r = requests.get("https://api.materialsproject.org/materials/summary/",
                                 params=p, headers=headers, timeout=60)
                if r.status_code == 200:
                    ok = True; break
                print(f"  MP HTTP {r.status_code}: {r.text[:120]}", flush=True)
            except Exception as e:
                print(f"  MP err {e}", flush=True)
            time.sleep(2 * (att + 1))
        if not ok:
            raise SystemExit("MP fetch gagal")
        data = r.json().get("data", [])
        rows.extend(data)
        skip += len(data)
        if len(data) < limit:
            break
        time.sleep(0.5)
    return rows

def do_fetch():
    print("== fetch energi total MP (815 entri Ca-P) ==")
    rows = mp_get({"elements": "Ca,P", "_fields": "material_id,energy_per_atom,formation_energy_per_atom"})
    json.dump(rows, open(f"{RAW}/mp_total_energy.json", "w"))
    print(f"   {len(rows)} entri, {sum(1 for r in rows if 'energy_per_atom' in r)} punya energy_per_atom")
    print("== fetch fase pesaings (6 sistem kimia) ==")
    all_e, seen = [], set()
    for cs in CHEMSYS:
        rows = mp_get({"chemsys": cs, "_fields": "material_id,formation_energy_per_atom,formula_pretty,elements,nsites"})
        n_new = 0
        for r in rows:
            if r["material_id"] not in seen:
                seen.add(r["material_id"]); all_e.append(r); n_new += 1
        print(f"   {cs}: {len(rows)} entri ({n_new} baru)")
    json.dump(all_e, open(f"{RAW}/mp_hull_entries.json", "w"))
    print(f"   total fase pesaings unik: {len(all_e)}")

# ---------------- mu fitting ----------------
def load_mp_fit_data():
    wp2 = json.load(open(f"{BASE}/wp2_dataset.json"))
    tot = {r["material_id"]: r.get("energy_per_atom") for r in json.load(open(f"{RAW}/mp_total_energy.json"))}
    entries = []
    for e in wp2["mp_entries"]:
        et = tot.get(e["material_id"])
        if et is None:
            continue
        entries.append({"material_id": e["material_id"], "counts": e["counts"],
                        "formation_e": e["formation_e"], "energy_per_atom": et})
    return entries

def fit_mu(entries, min_count=3):
    from collections import Counter
    cnt = Counter()
    for e in entries:
        for el in e["counts"]:
            cnt[el] += 1
    els = sorted([el for el, c in cnt.items() if c >= min_count])
    idx = {el: i for i, el in enumerate(els)}
    A, b = [], []
    for e in entries:
        n = sum(e["counts"].values())
        row = np.zeros(len(els))
        for el, k in e["counts"].items():
            if el in idx:
                row[idx[el]] = k / n
        A.append(row)
        b.append(e["energy_per_atom"] - e["formation_e"])
    A, b = np.array(A), np.array(b)
    mu, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    pred_form = np.array([e["energy_per_atom"] - A[i] @ mu for i, e in enumerate(entries)])
    actual = np.array([e["formation_e"] for e in entries])
    err = pred_form - actual
    return {"elements": els, "mu": {el: float(mu[idx[el]]) for el in els},
            "counts": {el: cnt[el] for el in els},
            "rmse": float(np.sqrt(np.mean(err ** 2))), "max_abs_err": float(np.max(np.abs(err))),
            "n_entries": len(entries)}

# ---------------- QE reading ----------------
def read_qe_total_energy(path):
    last = None
    for line in open(path):
        if line.lstrip().startswith("!") and "total energy" in line:
            last = float(line.split("=")[1].split()[0])
    if last is None:
        raise ValueError(f"tidak ada energi total di {path}")
    return last * RY_TO_EV  # eV per sel

def formation_energy_qe(e_total_ev, counts, mu):
    n = sum(counts.values())
    return (e_total_ev - sum(k * mu[el] for el, k in counts.items())) / n

# ---------------- hull ----------------
def load_hull_entries():
    """Muat entri hull MP (1.908 entri subsistem) sebagai dict.

    Mengembalikan dict reduced_formula -> (e_form_per_atom, Composition),
    didedup ke energi terendah per rumus (polimorf dasar). Entri unsur
    murni F/H/O TIDAK ada di berkas (gas tidak ada di ringkasan MP) — maka
    ditambahkan sintetis di sini (energi formasi unsur murni = 0 menurut
    definisi). Merge wp2 (815 entri Ca-P) SENGAJA DIHAPUS 2026-11-19:
    entri itu mencakup ~60 unsur substitusi tanpa entri terminalnya,
    sehingga PhaseDiagram gagal ("Missing terminal entries"); lagi pula
    subsistem 1.908 sudah mencakup semua sistem kimia kandidat putaran 1.
    """
    from pymatgen.core import Composition
    rows = json.load(open(f"{RAW}/mp_hull_subsystems.json"))
    best = {}
    for r in rows:
        comp = Composition(r["formula_pretty"])
        key = comp.reduced_formula
        e_per_atom = r["formation_energy_per_atom"]
        if key not in best or e_per_atom < best[key][0]:
            best[key] = (e_per_atom, comp)
    for el in ("F", "H", "O"):
        if el not in best:
            best[el] = (0.0, Composition(el))
    return best

def build_hull_for(els, hull):
    """Diagram fase KHUSUS untuk satu sistem kimia (unsur-unsur `els`).

    Hanya entri yang unsurnya subset dari `els` yang dipakai — fase di
    luar sistem itu tidak bisa menjadi produk dekomposisi kandidat.
    Ini juga menjaga dimensi hull tetap kecil (2–5 unsur), sehingga
    Qhull cepat dan stabil; hull 16 unsur sekaligus terlalu berat.
    """
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    els = set(els)
    entries = [ComputedEntry(comp, e_per_atom * comp.num_atoms)
               for e_per_atom, comp in hull.values()
               if set(str(e) for e in comp.elements).issubset(els)]
    return PhaseDiagram(entries)

# ---------------- calibrate ----------------
def do_calibrate():
    entries = load_mp_fit_data()
    fit = fit_mu(entries)
    print(f"== fit mu dari {fit['n_entries']} entri MP (SEKUNDER — hasilnya buruk, lihat bawah) ==")
    print(f"   RMSE energi formasi (fit vs MP): {fit['rmse']:.4f} eV/atom, maks |err| {fit['max_abs_err']:.4f}")
    print("   mu Zn positif (+4.71 eV) = tak-fisik: satu mu per unsur tidak bisa")
    print("   menyerap koreksi MP yang bergantung pada senyawa (koreksi oksida dsb.)")
    print("   + Ca dan P muncul di SEMUA entri (kolinear). Jadi mu-MP TIDAK dipakai.")
    ref = f"{BASE}/dft_r1/ref_beta_tcp.out"
    if not os.path.exists(ref):
        print(f"   BELUM ADA {ref} — salin ref.out beta-TCP dari VPS dulu.")
        return
    e_tot_qe = read_qe_total_energy(ref)  # eV per sel 13 atom
    # beta-TCP stabil di MP: mp-aaacqdgz (E_form -3.32566, di hull)
    tot = {r["material_id"]: r.get("energy_per_atom")
           for r in json.load(open(f"{RAW}/mp_total_energy.json"))}
    raw = json.load(open(f"{RAW}/mp_ca_p.json"))
    vasp = None
    for x in raw:
        if x.get("material_id") == "mp-aaacqdgz":
            vasp = tot["mp-aaacqdgz"] * 13  # eV per sel 13 atom
            mp_form = x["formation_energy_per_atom"]
    n = 13
    print("== kalibrasi beta-TCP (Ca3(PO4)2, 13 atom) — PERBANDINGAN LANGSUNG ==")
    print(f"   E total QE   (ref.out): {e_tot_qe:10.2f} eV/sel  ({e_tot_qe / n:9.3f} eV/atom)")
    print(f"   E total VASP (MP)     : {vasp:10.2f} eV/sel  ({vasp / n:9.3f} eV/atom)")
    print(f"   SELISIH ANTAR-KODE    : {e_tot_qe - vasp:10.2f} eV/sel  ({(e_tot_qe - vasp) / n:9.2f} eV/atom)")
    print(f"   E formasi MP          : {mp_form:.4f} eV/atom")
    print("   KESIMPULAN: energi total absolut QE ~714 eV/atom LEBIH NEGATIF daripada")
    print("   VASP untuk komposisi yang sama — skala nol pseudopotential berbeda.")
    print("   Mencampur keduanya menghasilkan energi formasi yang salah total.")
    print("   Maka energi formasi kandidat HARUS memakai referensi unsur QE sendiri")
    print("   (prepare_qe_ref.py: H2, O2, F2, Ca, K, Li, Ba, Zn, P hitam, C grafit).")
    json.dump({"fit_mu_mp": fit,
               "beta_tcp": {"e_total_qe_ev_per_cell": e_tot_qe,
                            "e_total_vasp_ev_per_cell": vasp,
                            "offset_ev_per_atom": (e_tot_qe - vasp) / n,
                            "e_form_mp": mp_form}},
              open(f"{BASE}/dft_r1/kalibrasi-beta-tcp.json", "w"), indent=2)
    print("   tersimpan: dft_r1/kalibrasi-beta-tcp.json")

# ---------------- mu QE dari job referensi ----------------
def do_mu_qe(refdir):
    import glob
    mu = {}
    # Pola glob HARUS mencakup batch 1 (ref_<El>.out) DAN batch 2
    # (ref2_<El>.out). "ref_*.out" saja TIDAK cocok dengan "ref2_Tb.out"
    # karena "_" di pola itu literal — bug ini membuat mu_Tb gagal masuk
    # 2026-11-19 (mu-qe berjalan tanpa error tapi tidak membaca apa-apa).
    paths = sorted(set(glob.glob(os.path.join(refdir, "ref_*.out")))
                   | set(glob.glob(os.path.join(refdir, "ref2_*.out"))))
    for path in paths:
        job = os.path.basename(path)[:-4]
        el = job.split("_")[1].rstrip("0123456789")  # molekul F2/H2/O2 -> unsur F/H/O
        e_tot = read_qe_total_energy(path)
        # jumlah atom unsur itu di sel: baca dari .in (nat) — molekul 2 atom, bulk 1-8
        nat = None
        for line in open(os.path.join(refdir, f"{job}.in")):
            if "nat " in line and "=" in line:
                nat = int(line.split("=")[1])
        mu[el] = e_tot / nat
        print(f"   {job}: E total {e_tot:.3f} eV / {nat} atom -> mu_{el} = {mu[el]:.4f} eV/atom")
    # MENGGABUNG dengan ref_qe_mu.json yang sudah ada (bukan menimpa):
    # batch 2 hanya berisi 8 unsur baru; 10 mu batch 1 harus tetap ada.
    out_path = f"{BASE}/dft_r1/ref_qe_mu.json"
    if os.path.exists(out_path):
        old = json.load(open(out_path))
        old.update(mu)
        mu = old
    json.dump(mu, open(out_path, "w"), indent=2)
    print(f"   tersimpan: {out_path} ({len(mu)} unsur)")
    # validasi silang: energi formasi beta-TCP dari mu QE vs MP
    ref = f"{BASE}/dft_r1/ref_beta_tcp.out"
    if os.path.exists(ref) and all(e in mu for e in ("Ca", "P", "O")):
        e_tot = read_qe_total_energy(ref)
        ef = formation_energy_qe(e_tot, {"Ca": 3, "P": 2, "O": 8}, mu)
        print(f"   VALIDASI beta-TCP: E formasi QE = {ef:.4f} eV/atom vs MP -3.3257 "
              f"-> selisih {ef - (-3.325657296804029):+.4f} eV/atom")

# ---------------- analyze (butuh .out + mu QE) ----------------
# Koreksi komposisi MP2020 (eV per atom anion) — MP2020Compatibility.yaml.
# Hanya anion yang tak-ambigu di senyawa kita: O (oksida; fosfat/apatit kita
# tidak punya ikatan O-O, jadi bukan peroksida/superoksida), F/Cl/I (halida).
# H TIDAK dikoreksi (H kita protik OH/HPO4, bukan hidrida); S/Si/Se TIDAK
# dikoreksi (butuh analisis bilangan oksidasi per senyawa).
CORR_ANION = {"O": -0.687, "F": -0.462, "Cl": -0.614, "I": -0.379}

def parse_counts_from_in(path):
    """Hitung komposisi sel dari blok ATOMIC_POSITIONS berkas .in."""
    import re
    counts = {}
    in_pos = False
    for line in open(path):
        s = line.strip()
        if s.startswith("ATOMIC_POSITIONS"):
            in_pos = True
            continue
        if in_pos:
            if not s:
                break
            tok = s.split()[0]
            if not re.fullmatch(r"[A-Z][a-z]?", tok):
                break
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        raise ValueError(f"tidak ada posisi atom di {path}")
    return counts

def do_analyze(outdir):
    import glob
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry
    mu_path = f"{BASE}/dft_r1/ref_qe_mu.json"
    if not os.path.exists(mu_path):
        print("ANALISIS BELUM BISA: referensi unsur QE (dft_r1/ref_qe_mu.json) belum ada.")
        print("Jalankan dulu: analisis_dft_r1.py --mu-qe dft_ref/")
        return
    mu = json.load(open(mu_path))
    outs = sorted(glob.glob(os.path.join(outdir, "*.out")))
    if not outs:
        print(f"Tidak ada berkas .out di {outdir}.")
        return
    print(f"== analisis {len(outs)} berkas .out di {outdir} ==")
    print("   Memuat entri hull MP (data_raw/mp_hull_subsystems.json)...", flush=True)
    hull = load_hull_entries()
    hull_els = set()
    for _, comp in hull.values():
        hull_els.update(str(el) for el in comp.elements)
    print(f"   Hull siap: {len(hull)} rumus unik, unsur: {sorted(hull_els)}", flush=True)
    pd_cache = {}
    results = []
    for opath in outs:
        job = os.path.basename(opath)[:-4]
        ipath = opath[:-4] + ".in"
        rec = {"job": job}
        if not os.path.exists(ipath):
            rec["status"] = "tanpa .in pendamping"
            results.append(rec)
            print(f"   {job}: {rec['status']}")
            continue
        text = open(opath, errors="ignore").read()
        if "JOB DONE" not in text:
            rec["status"] = "belum JOB DONE (masih jalan/gagal)"
            results.append(rec)
            print(f"   {job}: {rec['status']}")
            continue
        try:
            counts = parse_counts_from_in(ipath)
        except ValueError as e:
            rec["status"] = str(e)
            results.append(rec)
            print(f"   {job}: {rec['status']}")
            continue
        missing = [el for el in counts if el not in mu]
        if missing:
            rec["status"] = f"mu QE belum ada untuk {missing}"
            results.append(rec)
            print(f"   {job}: {rec['status']}")
            continue
        n = sum(counts.values())
        e_tot = read_qe_total_energy(opath)  # eV per sel (energi TERAKHIR = rileks)
        e_raw = (e_tot - sum(k * mu[el] for el, k in counts.items())) / n
        corr = sum(counts.get(a, 0) * c for a, c in CORR_ANION.items()) / n
        e_corr = e_raw + corr
        comp = Composition(counts)
        rec.update({"status": "ok", "formula": comp.reduced_formula,
                    "counts": dict(counts), "n_atoms": n,
                    "e_total_ev": round(e_tot, 3),
                    "e_form_qe_raw": round(e_raw, 4),
                    "e_form_qe_corr": round(e_corr, 4)})
        els = set(str(el) for el in comp.elements)
        if not els.issubset(hull_els):
            rec["e_above_hull"] = None
            rec["catatan_hull"] = f"hull belum mencakup unsur {sorted(els - hull_els)}"
        else:
            key = frozenset(els)
            if key not in pd_cache:
                try:
                    pd_cache[key] = build_hull_for(els, hull)
                except ValueError as e:
                    pd_cache[key] = None
                    rec["catatan_hull"] = f"hull sistem {sorted(els)} tidak bisa dibangun: {e}"
            pd = pd_cache[key]
            if pd is None:
                rec["e_above_hull"] = None
                rec.setdefault("catatan_hull",
                               f"hull sistem {sorted(els)} tidak bisa dibangun")
            else:
                entry = ComputedEntry(comp, e_corr * n)
                rec["e_above_hull"] = round(pd.get_e_above_hull(entry), 4)
        results.append(rec)
        h = rec["e_above_hull"]
        hs = f"{h:+.4f}" if h is not None else "n/a"
        print(f"   {job}: {rec['formula']} E_form(QE,corr)={e_corr:+.4f} "
              f"e_above_hull={hs} eV/atom")
    out_path = f"{BASE}/dft_r1/hasil_analisis_r1.json"
    json.dump(results, open(out_path, "w"), indent=2)
    ok = [r for r in results if r["status"] == "ok"]
    stab = [r for r in ok if r.get("e_above_hull") is not None and r["e_above_hull"] <= 0.05]
    print(f"   tersimpan: dft_r1/hasil_analisis_r1.json "
          f"({len(ok)} berlabel, {len(stab)} stabil/dalam 0,05 eV dari hull)")
    print("   LANGKAH BERIKUT: latih ulang surrogate dengan label baru ini, "
          "lalu tulis kesimpulan putaran 1.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--analyze", metavar="DIR", nargs="?", const="", default=None,
                    help="analisis pasangan <job>.out + <job>.in di DIR -> dft_r1/hasil_analisis_r1.json")
    ap.add_argument("--mu-qe", metavar="DIR", default=None,
                    help="baca berkas ref_*.out di DIR -> dft_r1/ref_qe_mu.json")
    a = ap.parse_args()
    if a.fetch:
        do_fetch()
    elif a.calibrate:
        do_calibrate()
    elif a.analyze is not None:
        if not a.analyze:
            print("Pakai: analisis_dft_r1.py --analyze DIR  (DIR berisi pasangan <job>.out + <job>.in)")
        else:
            do_analyze(a.analyze)
    elif a.mu_qe:
        do_mu_qe(a.mu_qe)
    else:
        ap.print_help()
