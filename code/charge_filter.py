#!/usr/bin/env python3
"""Filter keseimbangan muatan (charge-balance) untuk 30 kandidat putaran 2 PRGF.

Ide: rumus suatu senyawa netral harus bermuatan bersih nol bila tiap unsur
diberi keadaan oksidasi standarnya. Filter ini menolak kandidat yang
rumusnya tidak setimbang muatan — tanpa biaya DFT, tanpa GNN, hanya dari
rumus yang sudah ada.

Sumber data: aturan_seleksi_v2.json (30 kandidat draf putaran 2 — rumus,
bio_lolos, hull_LCB, sd, sumber; sumber yang sama dengan aturan_seleksi_v2.py,
yaitu dft_r2/manifest.json) dan dft_labels_r2_train.json (5 label DFT
dengan counts unsur + E_above_hull).

Validasi (WAJIB lolos sebelum 30 kandidat dihitung): hitungan manual teman
dari Tabel 3 (keadaan oksidasi standar) — r2_01 LOLOS, r2_16 LOLOS,
r2_17 DITOLAK, r2_19 DITOLAK, r2_24 DITOLAK. Bila skrip menghasilkan lain,
skrip berhenti dengan exit code 1 dan TIDAK menulis hasil 30 kandidat.

Aturan lolos: muatan bersih == 0 (bilangan bulat, eksak).

Keluaran: charge_filter_r2.json (per kandidat + ringkasan + sensitivitas).
"""
import json, os, re, sys
from collections import OrderedDict

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- tabel oksidasi
# Keadaan oksidasi primer + alasan. Unsur bervalensi berubah diberi satu
# pilihan primer; alternatifnya diuji di analisis sensitivitas.
# Oksigen diasumsikan O2- di semua rumus (semua kandidat adalah oksida/
# fosfat; tidak ada peroksida/superoaksida di antara 30 rumus ini).
OXIDATION = {
    # unsur dasar Ca-P-O-H + halogen
    "H":  (+1, "proton; gugus OH/HO3 dalam fosfat"),
    "O":  (-2, "oksida; semua kandidat adalah oksida/fosfat"),
    "F":  (-1, "fluorida; substitusi situs OH"),
    "Cl": (-1, "klorida; substitusi situs OH"),
    # kation alkali / alkali tanah (satu keadaan oksidasi)
    "Ca": (+2, "alkali tanah; situs utama Ca"),
    "Mg": (+2, "substitusi Ca2+ (ion terverifikasi)"),
    "Sr": (+2, "substitusi Ca2+ (ion terverifikasi)"),
    "Ba": (+2, "alkali tanah; tidak muncul di 30 kandidat"),
    "Zn": (+2, "substitusi Ca2+ (ion terverifikasi)"),
    "Na": (+1, "alkali; substitusi minor situs Ca"),
    "K":  (+1, "alkali; tidak muncul di 30 kandidat"),
    "Li": (+1, "alkali; tidak muncul di 30 kandidat"),
    "Cs": (+1, "alkali"),
    # pembentuk polianion
    "P":  (+5, "fosfat PO4(3-)"),
    "Si": (+4, "silikat SiO4(4-) (ion terverifikasi)"),
    "B":  (+3, "borat (BO3/BO4)"),
    "Al": (+3, "golongan 13; satu keadaan oksidasi di oksida"),
    "In": (+3, "golongan 13, seperti Al"),
    "Hf": (+4, "golongan 4; satu keadaan oksidasi di oksida"),
    # bervalensi berubah — pilihan primer + alasan
    "Mn": (+2, "Mn2+ substitusi situs Ca2+ (ion terverifikasi; literatur HAp)"),
    "Ce": (+3, "Ce3+ lantanida; keadaan umum substitusi situs Ca"),
    "Tb": (+3, "Tb3+ lantanida; DIPAKSA oleh validasi: r2_16 hanya lolos bila Tb=+3"),
    "Eu": (+3, "Eu3+ lantanida; keadaan umum dalam fosfat"),
    "Sm": (+3, "Sm3+ lantanida; keadaan umum substitusi situs Ca"),
    "Nd": (+3, "Nd3+ lantanida; keadaan umum substitusi situs Ca"),
    "C":  (+4, "karbonat CO3(2-) — bentuk C dalam mineral tulang/fosfat"),
    # kalkogen: S/Se sebagai sulfida/selenida (-2), BUKAN sulfat (+6):
    # jumlah O di rumus-rumus ini persis 4xP (oksigen fosfat saja) sehingga
    # tidak ada O tersisa untuk gugus SO4/SeO4; S/Se menggantikan anion
    # (analog 2 OH- -> 1 S2-, seperti Ca10(PO4)6S).
    "S":  (-2, "sulfida S2-; O rumus hanya cukup untuk gugus PO4, bukan SO4"),
    "Se": (-2, "selenida Se2-; analog S (O rumus hanya cukup untuk PO4)"),
}
OXI = {el: v[0] for el, v in OXIDATION.items()}

# Alternatif untuk analisis sensitivitas: unsur -> keadaan oksidasi alternatif.
ALTERNATIVES = {
    "S":  [6],    # sulfat
    "Se": [6],    # selenat
    "Eu": [2],    # Eu2+
    "Mn": [3, 4], # Mn3+/Mn4+
    "Ce": [4],    # Ce4+
    "C":  [-4],   # karbida
    "Sm": [2],    # Sm2+
    "Nd": [2],    # Nd2+
    # Tb sengaja TIDAK diuji alternatifnya: Tb=+4 akan menolak r2_16 dan
    # melanggar validasi (hitungan manual teman) — lihat laporan.
}

# ---------------------------------------------------------------- parser rumus
TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(formula):
    """'Ca14F3MnO36P9' -> {'Ca': 14, 'F': 3, 'Mn': 1, 'O': 36, 'P': 9}."""
    counts = {}
    for el, n in TOKEN.findall(formula):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts

def net_charge(counts, oxi=None):
    oxi = oxi or OXI
    total = 0
    unknown = []
    for el, n in counts.items():
        if el not in oxi:
            unknown.append(el)
        else:
            total += n * oxi[el]
    return total, unknown

def cp_interval(x, n):
    """Clopper-Pearson 95% eksak untuk x keberhasilan dari n percobaan."""
    from scipy.stats import beta
    if n == 0:
        return None
    lo = 0.0 if x == 0 else beta.ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(0.975, x + 1, n - x)
    return (round(float(lo), 2), round(float(hi), 2))

def backtest(selected, labelled):
    """selected: set job yang dipilih; labelled: {job: stabil_bool}."""
    tp = sum(1 for j, s in labelled.items() if j in selected and s)
    fp = sum(1 for j, s in labelled.items() if j in selected and not s)
    fn = sum(1 for j, s in labelled.items() if j not in selected and s)
    tn = sum(1 for j, s in labelled.items() if j not in selected and not s)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": prec, "recall": rec,
            "precision_CI95": cp_interval(tp, tp + fp),
            "recall_CI95": cp_interval(tp, tp + fn)}

# ---------------------------------------------------------------- muat data
sel = json.load(open(f"{BASE}/aturan_seleksi_v2.json"))
cands = sel["kandidat"]
assert len(cands) == 30, f"harus 30 kandidat, dapat {len(cands)}"

labels_file = json.load(open(f"{BASE}/dft_labels_r2_train.json"))
lab_records = {r["job"]: r for r in labels_file["records"]}
# stabil = E_above_hull <= 0,05 (definisi yang sama dengan aturan_seleksi_v2.py)
labelled = {j: (r["e_above_hull"] <= 0.05) for j, r in lab_records.items()}
assert len(labelled) == 5, f"harus 5 label, dapat {len(labelled)}"

# ---------------------------------------------------------------- LANGKAH 2: validasi 5 label
EXPECTED = {"r2_01_Ca3O8P2": True, "r2_16_Ca2O6PTb": True,
            "r2_17_Ca2CeO8P": False, "r2_19_Ca2H2O6P": False,
            "r2_24_Al2CaO6P": False}
print("== VALIDASI 5 label (hitungan manual teman) ==")
validation = {}
ok = True
for job, expect_pass in EXPECTED.items():
    counts = lab_records[job]["counts"]
    formula = lab_records[job]["reduced_formula"]
    q, unk = net_charge(counts)
    assert not unk, f"unsur tak dikenal di {job}: {unk}"
    passed = (q == 0)
    validation[job] = {"formula": formula, "counts": counts,
                       "net_charge": q, "lolos": passed,
                       "diharapkan_lolos": expect_pass,
                       "stabil_dft": labelled[job]}
    status = "COCOK" if passed == expect_pass else "SELISIH!"
    if passed != expect_pass:
        ok = False
    print(f"  {job} {formula:14s} muatan {q:+d} -> "
          f"{'LOLOS' if passed else 'DITOLAK'} (diharapkan "
          f"{'LOLOS' if expect_pass else 'DITOLAK'}) {status}")
if not ok:
    print("\nVALIDASI GAGAL — berhenti, tidak menghitung 30 kandidat.")
    print("Perbedaan asumsi oksidasi dengan hitungan manual teman harus "
          "diselesaikan dulu. Rincian di atas (rumus + muatan bersih).")
    json.dump({"validasi": validation, "validasi_lolos": False},
              open(f"{BASE}/charge_filter_r2.json", "w"), indent=2)
    sys.exit(1)
print("Validasi COCOK 5/5 — lanjut ke 30 kandidat.\n")

# ---------------------------------------------------------------- LANGKAH 3: 30 kandidat
def r2_select(r):
    return r["hull_lcb"] <= 0.05 and (r["source"] == "MP" or r["gnn_sd"] <= 0.10)

def r5_select(r):
    return r["bio_lolos"] and r2_select(r)

rows = []
for r in cands:
    counts = parse_formula(r["formula"])
    q, unk = net_charge(counts)
    assert not unk, f"unsur tak dikenal di {r['job']}: {unk}"
    rows.append({"job": r["job"], "formula": r["formula"],
                 "counts": counts, "net_charge": q, "lolos_muatan": q == 0,
                 "bio_lolos": r["bio_lolos"],
                 "lolos_R2": r2_select(r), "lolos_R5": r5_select(r),
                 "source": r["source"], "tier": r["tier"],
                 "dft_hull": r.get("dft_hull"),
                 "stabil_dft": r.get("stabil_dft")})

# verifikasi aturan R2/R5 hasil hitung-ulang cocok dengan ringkasan tersimpan:
# (a) pilihan tak berlabel sama dengan daftar "sisa"; (b) backtest pada
# 5 label sama TP/FP/FN-nya.
calc_r2 = {r["job"] for r in rows if r["lolos_R2"]}
calc_r5 = {r["job"] for r in rows if r["lolos_R5"]}
assert calc_r2 - set(labelled) == set(sel["aturan"]["R2"]["sisa"]), \
    f"R2 tak-berlabel {sorted(calc_r2 - set(labelled))} != tersimpan {sel['aturan']['R2']['sisa']}"
assert calc_r5 - set(labelled) == set(sel["aturan"]["R5"]["sisa"]), \
    f"R5 tak-berlabel {sorted(calc_r5 - set(labelled))} != tersimpan {sel['aturan']['R5']['sisa']}"
for rule, calc in (("R2", calc_r2), ("R5", calc_r5)):
    bt = backtest(calc, labelled)
    st = sel["aturan"][rule]
    assert (bt["TP"], bt["FP"], bt["FN"]) == (st["TP"], st["FP"], st["FN"]), \
        f"backtest {rule} hitung-ulang {(bt['TP'], bt['FP'], bt['FN'])} != tersimpan {(st['TP'], st['FP'], st['FN'])}"
print(f"Aturan R2/R5 hasil hitung-ulang cocok dengan aturan_seleksi_v2.json "
      f"(R2: {len(calc_r2)}, R5: {len(calc_r5)}).")

charge_sel = {r["job"] for r in rows if r["lolos_muatan"]}
and_sel = {r["job"] for r in rows if r["bio_lolos"] and r["lolos_muatan"]}
r7_sel = {r["job"] for r in rows if r["lolos_R5"] and r["lolos_muatan"]}
unlab = [r for r in rows if r.get("stabil_dft") is None]  # kunci selalu ada (None bila tak berlabel)
assert len(unlab) == 25

bt_charge = backtest(charge_sel, labelled)
bt_and = backtest(and_sel, labelled)
bt_r7 = backtest(r7_sel, labelled)
bt_bio = backtest({r["job"] for r in rows if r["bio_lolos"]}, labelled)

n_charge_unlab = sum(1 for r in unlab if r["lolos_muatan"])
n_and_unlab = sum(1 for r in unlab if r["bio_lolos"] and r["lolos_muatan"])
# Irisan dengan R2 — hitung pada 9 pilihan R2 yang TAK BERLABEL (daftar
# "sisa"), sesuai pertanyaan; pilihan R2 yang berlabel dilaporkan terpisah.
r2_sisa = set(sel["aturan"]["R2"]["sisa"])
r2_inter = sorted(r2_sisa & charge_sel)
r2_labelled_picks = sorted(j for j in labelled if j in calc_r2)
r2_labelled_charge = {j: next(r for r in rows if r["job"] == j)["lolos_muatan"]
                      for j in r2_labelled_picks}
r5_prospektif = sorted(set(sel["aturan"]["R5"]["sisa"]))
r5_charge = {j: next(r for r in rows if r["job"] == j)["lolos_muatan"]
             for j in r5_prospektif}

print("\n== HASIL 30 kandidat ==")
print(f"Lolos filter muatan: {len(charge_sel)}/30 "
      f"(label: {sum(1 for j in labelled if j in charge_sel)}/5, "
      f"tak berlabel: {n_charge_unlab}/25)")
print(f"AND (biologi + muatan): {len(and_sel)}/30 (tak berlabel: {n_and_unlab}/25)")
print(f"R7 (R5 + muatan): {len(r7_sel)}/30 -> {sorted(r7_sel)}")
print(f"R2 tak-berlabel yang lolos muatan: {len(r2_inter)}/9: {r2_inter}")
print(f"R2 berlabel yang lolos muatan: "
      f"{sum(r2_labelled_charge.values())}/{len(r2_labelled_picks)}: "
      f"{r2_labelled_charge}")
print(f"R5 prospektif vs muatan: {r5_charge}")
print(f"\nBacktest muatan saja: {bt_charge}")
print(f"Backtest AND bio+muatan: {bt_and}")
print(f"Backtest R7 (R5+muatan): {bt_r7}")
print(f"Backtest biologi saja (kontrol): {bt_bio}")

# ---------------------------------------------------------------- sensitivitas
print("\n== SENSITIVITAS (ganti satu unsur ke keadaan alternatif) ==")
sensitivity = {}
for el, alts in ALTERNATIVES.items():
    for alt in alts:
        oxi2 = dict(OXI); oxi2[el] = alt
        flips = []
        for r in rows:
            if el not in r["counts"]:
                continue
            q2, _ = net_charge(r["counts"], oxi2)
            if (q2 == 0) != r["lolos_muatan"]:
                flips.append(r["job"])
        # dampak pada metrik utama
        ch2 = {r["job"] for r in rows
               if net_charge(r["counts"], oxi2)[0] == 0}
        and2 = {r["job"] for r in rows
                if r["bio_lolos"] and net_charge(r["counts"], oxi2)[0] == 0}
        r72 = {r["job"] for r in rows
               if r["lolos_R5"] and net_charge(r["counts"], oxi2)[0] == 0}
        bt2 = backtest(ch2, labelled)
        bt_and2 = backtest(and2, labelled)
        sensitivity[f"{el}:{OXI[el]}->{alt}"] = {
            "kandidat_berubah": sorted(flips),
            "lolos_muatan_total": len(ch2),
            "AND_tak_berlabel": sum(1 for r in unlab
                                    if r["job"] in and2),
            "R7": sorted(r72),
            "backtest_muatan": {k: bt2[k] for k in ("TP", "FP", "TN", "FN")},
            "backtest_AND": {k: bt_and2[k] for k in ("TP", "FP", "TN", "FN")}}
        print(f"  {el} {OXI[el]:+d} -> {alt:+d}: berubah {sorted(flips)}; "
              f"lolos {len(ch2)}/30; AND-25 {sensitivity[f'{el}:{OXI[el]}->{alt}']['AND_tak_berlabel']}; "
              f"R7 {sorted(r72)}; backtest muatan TP/FP {bt2['TP']}/{bt2['FP']}; "
              f"backtest AND TP/FP {bt_and2['TP']}/{bt_and2['FP']}")

# ---------------------------------------------------------------- tulis JSON
out = {
    "metode": "Filter keseimbangan muatan: muatan bersih rumus (keadaan oksidasi "
              "standar) harus == 0. Sumber rumus: aturan_seleksi_v2.json "
              "(= dft_r2/manifest.json); label validasi: dft_labels_r2_train.json.",
    "tabel_oksidasi": {el: {"muatan": v[0], "alasan": v[1]}
                       for el, v in OXIDATION.items()},
    "validasi_5_label": validation,
    "validasi_lolos": True,
    "kandidat": [{k: r[k] for k in ("job", "formula", "counts", "net_charge",
                                    "lolos_muatan", "bio_lolos", "lolos_R2",
                                    "lolos_R5", "source", "tier",
                                    "dft_hull", "stabil_dft")}
                 for r in rows],
    "ringkasan": {
        "lolos_muatan_30": len(charge_sel),
        "lolos_muatan_tak_berlabel_25": n_charge_unlab,
        "AND_bio_muatan_30": len(and_sel),
        "AND_bio_muatan_tak_berlabel_25": n_and_unlab,
        "AND_bio_muatan_tak_berlabel": sorted(
            j for j in and_sel if j not in labelled),
        "R7_R5_plus_muatan": sorted(r7_sel),
        "R7_R5_plus_muatan_tak_berlabel": sorted(
            j for j in r7_sel if j not in labelled),
        "R2_lolos_muatan": r2_inter,
        "R2_lolos_muatan_jumlah": f"{len(r2_inter)}/9 (tak berlabel)",
        "R2_berlabel_lolos_muatan": r2_labelled_charge,
        "R2_lolos_muatan_total_termasuk_label": (
            f"{len(r2_inter) + sum(r2_labelled_charge.values())}/12"),
        "R5_prospektif_vs_muatan": r5_charge,
        "backtest_muatan_saja": bt_charge,
        "backtest_AND_bio_muatan": bt_and,
        "backtest_R7": bt_r7,
        "backtest_biologi_saja_kontrol": bt_bio},
    "sensitivitas": sensitivity,
}
json.dump(out, open(f"{BASE}/charge_filter_r2.json", "w"), indent=2)
print("\nWrote charge_filter_r2.json")
