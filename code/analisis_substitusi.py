#!/usr/bin/env python3
"""Hitung energi substitusi putaran 3 dari .out DFT yang sudah JOB DONE.

  Situs Ca : E_sub = E(doped) - E(host) - mu(dopan) + mu(Ca)   [per atom dopan]
  Si-HA    : E_sub = E(doped+vakOH) - E(host) - mu(Si) + mu(P) + mu(H) + mu(O)
             (vakansi OH dihitung terhadap reservoir H+O — pilihan ini
             didokumentasikan di rencana-putaran3-substitusi.md §4 karena
             ia memengaruhi angka E_sub Si.)
  Turunan  : dV/V vs host (dari 'unit-cell volume' terakhir di .out).

Acuan host:
  HA  : dft_r2/hasil/r2_11_Ca5HO13P3.out bila ada, kalau tidak
        dft_r3/r3_host_ha.out (keduanya struktur identik).
  TCP : dft_r3/r3_host_tcp.out (sel 52-atom; r2_01 13-atom TIDAK dipakai).
mu dari dft_r1/ref_qe_mu.json. Bila mu dopan belum ada (Mg/Mn = batch 3),
job itu dilaporkan 'mu belum ada', bukan ditebak.

Jalankan: .venv-wp2/bin/python analisis_substitusi.py
Keluaran: dft_r3/hasil_substitusi.json + tabel tercetak.
"""
import json, os, re
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(BASE, "dft_r3")
RY_TO_EV = 13.605693122994

JOB_DONE = "JOB DONE"


def read_total_energy(out_path):
    """Energi total TERAKHIR (Ry) dari baris '! ... total energy' + JOB DONE."""
    last = None
    done = False
    with open(out_path, errors="ignore") as f:
        for line in f:
            if JOB_DONE in line:
                done = True
            if line.lstrip().startswith("!"):
                m = re.search(r"total energy\s+=\s+(-?\d+\.\d+)\s+Ry", line)
                if m:
                    last = float(m.group(1))
    return last, done


def read_volume(out_path):
    """Volume sel TERAKHIR (Bohr^3) dari 'unit-cell volume'."""
    last = None
    with open(out_path, errors="ignore") as f:
        for line in f:
            m = re.search(r"unit-cell volume\s+=\s+(-?\d+\.\d+)", line)
            if m:
                last = float(m.group(1))
    return last


def main():
    man = json.load(open(os.path.join(R3, "manifest.json")))
    mu = json.load(open(os.path.join(BASE, "dft_r1", "ref_qe_mu.json")))
    jobs = {j["job"]: j for j in man["jobs"]}

    # --- energi host ---
    host_e, host_v = {}, {}
    ha_candidates = [os.path.join(BASE, "dft_r2", "hasil", "r2_11_Ca5HO13P3.out"),
                     os.path.join(R3, "r3_host_ha.out")]
    for p in ha_candidates:
        if os.path.exists(p):
            e, done = read_total_energy(p)
            if done and e is not None:
                host_e["HA"] = e * RY_TO_EV
                host_v["HA"] = read_volume(p)
                print(f"Host HA dari {os.path.basename(p)}: {host_e['HA']:.4f} eV")
                break
    p = os.path.join(R3, "r3_host_tcp.out")
    if os.path.exists(p):
        e, done = read_total_energy(p)
        if done and e is not None:
            host_e["TCP"] = e * RY_TO_EV
            host_v["TCP"] = read_volume(p)
            print(f"Host TCP dari r3_host_tcp.out: {host_e['TCP']:.4f} eV")

    results = []
    for j in man["jobs"]:
        job = j["job"]
        if j["stage"] == "host":
            continue
        out = os.path.join(R3, f"{job}.out")
        rec = {"job": job, "stage": j["stage"], "host": j["host"],
               "site": j["site"], "dopant": j["dopant"],
               "fraction": j["fraction"]}
        if not os.path.exists(out):
            rec["status"] = "belum jalan"
            results.append(rec)
            continue
        e_ry, done = read_total_energy(out)
        if not done or e_ry is None:
            rec["status"] = "belum JOB DONE"
            results.append(rec)
            continue
        e_ev = e_ry * RY_TO_EV
        vol = read_volume(out)
        rec["E_total_eV"] = round(e_ev, 4)
        rec["E_per_atom_eV"] = round(e_ev / j["nsites"], 4)
        if host_v.get(j["host"]) and vol:
            rec["dV_V_pct"] = round(100.0 * (vol - host_v[j["host"]]) / host_v[j["host"]], 2)
        host = host_e.get(j["host"])
        dop = j["dopant"]
        if host is None:
            rec["status"] = "host belum ada"
        elif dop not in mu:
            rec["status"] = f"mu {dop} belum ada (batch 3)"
        elif j["stage"] == "A":
            if "Ca" not in mu:
                rec["status"] = "mu Ca belum ada"
            else:
                rec["E_sub_eV"] = round(e_ev - host - mu[dop] + mu["Ca"], 4)
                rec["status"] = "berlabel"
        elif j["job"] == "r3_21":
            missing = [x for x in ("Si", "P", "H", "O") if x not in mu]
            if missing:
                rec["status"] = f"mu belum ada: {missing}"
            else:
                rec["E_sub_eV"] = round(
                    e_ev - host - mu["Si"] + mu["P"] + mu["H"] + mu["O"], 4)
                rec["status"] = "berlabel"
        results.append(rec)

    n_lab = sum(1 for r in results if r.get("status") == "berlabel")
    print(f"\nE_sub putaran 3: {n_lab}/{len(results)} berlabel")
    for r in results:
        if r.get("status") == "berlabel":
            print(f"  {r['job']} {r['host']} {r['site']} {r['dopant']}: "
                  f"E_sub={r['E_sub_eV']:+.4f} eV  dV/V={r.get('dV_V_pct', '?')}%")
        else:
            print(f"  {r['job']} {r['host']} {r['site']} {r['dopant']}: {r['status']}")
    json.dump({"host_e_eV": host_e, "results": results},
              open(os.path.join(R3, "hasil_substitusi.json"), "w"), indent=2)
    print("tersimpan: dft_r3/hasil_substitusi.json")


if __name__ == "__main__":
    main()
