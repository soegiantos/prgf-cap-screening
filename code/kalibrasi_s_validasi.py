#!/usr/bin/env python3
"""Kalibrasi s ensemble GNN pada split VALIDASI (bukan test) — PRGF baru.

Latar: s=1,85 difit pada split TEST Ca-P (n=115) sehingga cakupan
+-1 sigma mencapai 68% — bocor (in-sample), sudah diungkapkan di
Limitations naskah. Perbaikan (permintaan teman pemilik, 2026-11-22):
fit s pada split validasi (n=127), laporkan cakupan pada test split.

Skrip ini:
  1. Memuat 5 bobot fine-tune (gnn_seed_100..104.pt) + wp2_dataset.json.
  2. Menghitung prediksi ensemble (mean + sd) untuk val (127) dan
     test (115) — arsitektur + build_graph + collate DISALIN dari
     pretrain_gnn.py (tidak diimpor: modul itu memuat dataset pretrain
     saat import, dan berkasnya tidak ada di VM ini).
  3. Pemeriksaan kewarasan: prediksi test hasil skrip ini HARUS
     mereproduksi wp2_gnn_pretrained.json (per_entry) dalam toleransi
     numerik; bila tidak, berhenti.
  4. Memindai s (langkah 0,01) pada validasi -> s' (cakupan terdekat
     0,68); melaporkan cakupan test dengan s' (out-of-sample) dan
     dengan s=1,85 (pembanding).
  5. Menghitung ulang hull_LCB ke-30 kandidat dengan s' (via pymatgen,
     jalur yang sama dengan aturan_seleksi_v2.py) dan membandingkan:
     pool R2, backtest R0/R2/R5, seleksi prospektif R5/R6/R6b vs yang
     dibekukan.
Keluaran: kalibrasi_s_validasi.json + kalibrasi_s_validasi.md.
"""
import json, math, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("WP2_THREADS", "4")))
DEVICE = torch.device("cpu")

MAX_NBR, RADIUS = 12, 8.0
BATCH = 8


def build_graph(e, elidx):
    lat = np.array(e["lattice"], float)
    frac = np.array(e["frac_coords"], float)
    els = np.array([elidx[x] for x in e["elements"]], dtype=np.int64)
    n = len(els)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    nbr_idx = np.zeros((n, MAX_NBR), dtype=np.int64)
    nbr_dist = np.zeros((n, MAX_NBR), dtype=np.float32)
    for a0 in range(0, n, 64):
        a1 = min(a0 + 64, n)
        diff = frac[None, :, None, :] + shifts[None, None, :, :] - frac[a0:a1, None, None, :]
        cart = diff @ lat
        dist = np.linalg.norm(cart, axis=-1)
        blk = np.arange(a1 - a0)
        dist[blk, np.arange(a0, a1), 13] = np.inf
        flat = dist.reshape(a1 - a0, -1)
        k = min(MAX_NBR, flat.shape[1])
        sel = np.argpartition(flat, k - 1, axis=1)[:, :k]
        sel = sel[np.arange(a1 - a0)[:, None], np.argsort(np.take_along_axis(flat, sel, axis=1), axis=1)]
        dsel = np.take_along_axis(flat, sel, axis=1)
        ok = dsel < RADIUS
        nbr_idx[a0:a1] = np.where(ok, (sel // 27), np.arange(a0, a1)[:, None])
        nbr_dist[a0:a1] = np.where(ok, dsel, 0.0)
    return {"atom": els, "nbr_idx": nbr_idx, "nbr_dist": nbr_dist,
            "y": float(e["formation_e"]), "split": e["split"], "mid": e["material_id"]}


class DS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    max_n = max(len(b["atom"]) for b in batch)
    B = len(batch)
    atom = np.zeros((B, max_n), dtype=np.int64)
    nbr_idx = np.zeros((B, max_n, MAX_NBR), dtype=np.int64)
    nbr_dist = np.zeros((B, max_n, MAX_NBR), dtype=np.float32)
    mask = np.zeros((B, max_n), dtype=np.float32)
    y = np.zeros(B, dtype=np.float32)
    for i, b in enumerate(batch):
        n = len(b["atom"])
        atom[i, :n] = b["atom"]
        nbr_idx[i, :n] = b["nbr_idx"]
        nbr_dist[i, :n] = b["nbr_dist"]
        mask[i, :n] = 1.0
        y[i] = b["y"]
    return (torch.from_numpy(atom), torch.from_numpy(nbr_idx),
            torch.from_numpy(nbr_dist), torch.from_numpy(mask), torch.from_numpy(y))


class GNN(nn.Module):
    def __init__(self, n_elem, emb=64, n_conv=3, n_gauss=40):
        super().__init__()
        self.emb = nn.Embedding(n_elem + 1, emb, padding_idx=0)
        self.n_gauss = n_gauss
        self.centers = torch.linspace(0, RADIUS, n_gauss)
        self.conv = nn.ModuleList()
        for _ in range(n_conv):
            self.conv.append(nn.Linear(2 * emb + n_gauss, 2 * emb))
        self.out1 = nn.Linear(emb, 64)
        self.out2 = nn.Linear(64, 1)
        self.act = nn.Softplus()

    def forward(self, atom, nbr_idx, nbr_dist, mask):
        h = self.emb(atom)
        g = torch.exp(-((nbr_dist.unsqueeze(-1) - self.centers.to(nbr_dist.device)) ** 2) / 0.25)
        B, N, K = nbr_idx.shape
        for lin in self.conv:
            h_nbr = h.gather(1, nbr_idx.reshape(B, N * K, 1).expand(-1, -1, h.shape[-1])).reshape(B, N, K, -1)
            z = torch.cat([h.unsqueeze(2).expand(-1, -1, K, -1), h_nbr, g], dim=-1)
            out = lin(z)
            gate, msg = torch.sigmoid(out[..., :h.shape[-1]]), self.act(out[..., h.shape[-1]:])
            h = h + (gate * msg).sum(dim=2) / K
        h = h * mask.unsqueeze(-1)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.out2(self.act(self.out1(pooled))).squeeze(-1)


def predict(model, subset):
    model.eval()
    dl = DataLoader(DS(subset), batch_size=64, collate_fn=collate)
    ps, ys = [], []
    with torch.no_grad():
        for atom, ni, nd, mk, y in dl:
            ps.append(model(atom, ni, nd, mk).cpu().numpy())
            ys.append(y.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)


def coverage(err, sd, s):
    return float(np.mean(err <= s * sd))


def main():
    ds = json.load(open(f"{BASE}/wp2_dataset.json"))
    cap = ds["mp_entries"]
    # --- muat bobot, pakai daftar unsur dari checkpoint (otoritatif) ---
    cks = []
    for s in range(100, 105):
        ck = torch.load(f"{BASE}/gnn_seed_{s}.pt", map_location="cpu", weights_only=False)
        assert ck["phase"] == "finetuned" and ck["seed"] == s
        cks.append(ck)
    elements = cks[0]["elements"]
    assert all(ck["elements"] == elements for ck in cks), "daftar unsur antar seed berbeda"
    elidx = {el: i + 1 for i, el in enumerate(elements)}
    print(f"Unsur: {len(elements)}; bobot 5 seed OK (finetuned)")

    graphs = [build_graph(e, elidx) for e in cap]
    va = [g for g in graphs if g["split"] == "val"]
    te = [g for g in graphs if g["split"] == "test"]
    print(f"Graf Ca-P: val {len(va)}, test {len(te)}")
    assert len(va) == 127 and len(te) == 115

    preds_va, preds_te, y_va, y_te = [], [], None, None
    for ck in cks:
        model = GNN(len(elements)).to(DEVICE)
        model.load_state_dict(ck["state_dict"])
        p_va, y_va = predict(model, va)
        p_te, y_te = predict(model, te)
        preds_va.append(p_va)
        preds_te.append(p_te)
    P_va = np.stack(preds_va)
    P_te = np.stack(preds_te)
    mean_va, sd_va = P_va.mean(axis=0), P_va.std(axis=0)
    mean_te, sd_te = P_te.mean(axis=0), P_te.std(axis=0)
    err_va = np.abs(mean_va - y_va)
    err_te = np.abs(mean_te - y_te)

    # --- pemeriksaan kewarasan: reproduksi wp2_gnn_pretrained.json ---
    stored = json.load(open(f"{BASE}/wp2_gnn_pretrained.json"))
    te_mids = [g["mid"] for g in te]
    stored_map = {r["material_id"]: r for r in stored["per_entry"]}
    assert set(stored_map) == set(te_mids), "ID test tidak cocok dengan berkas tersimpan"
    d_mean = max(abs(mean_te[i] - stored_map[m]["pred_mean"]) for i, m in enumerate(te_mids))
    d_sd = max(abs(sd_te[i] - stored_map[m]["pred_sd"]) for i, m in enumerate(te_mids))
    print(f"Kewarasan test vs tersimpan: max |d mean| = {d_mean:.2e}, max |d sd| = {d_sd:.2e}")
    assert d_mean < 1e-4 and d_sd < 1e-4, "prediksi test TIDAK mereproduksi berkas tersimpan"
    cov_te_185_stored = stored["calibration"]["coverage_1sd_scaled"]
    cov_te_185 = coverage(err_te, sd_te, 1.85)
    print(f"Kewarasan cakupan test s=1,85: hitung {cov_te_185:.4f} vs tersimpan {cov_te_185_stored:.4f}")
    assert abs(cov_te_185 - cov_te_185_stored) < 1e-9

    # --- pindai s pada VALIDASI (langkah 0,01) ---
    grid = np.arange(0.50, 6.001, 0.01)
    covs = [coverage(err_va, sd_va, s) for s in grid]
    best_i = int(np.argmin([abs(c - 0.68) for c in covs]))
    s_prime = float(grid[best_i])
    cov_va_prime = covs[best_i]
    cov_va_185 = coverage(err_va, sd_va, 1.85)
    cov_te_prime = coverage(err_te, sd_te, s_prime)
    # tepi eksak fungsi tangga: rasio |err|/sd terurut ke-86 (86/127 paling dekat 0,68)
    ratios = np.sort(err_va / sd_va)
    s_edge = float(ratios[85])  # s terkecil dengan cakupan >= 86/127
    print(f"\nValidasi n=127: cakupan s=1,85 -> {cov_va_185:.4f} ({int(round(cov_va_185*127))}/127)")
    print(f"s' (pindai 0,01) = {s_prime:.2f}, cakupan validasi = {cov_va_prime:.4f} ({int(round(cov_va_prime*127))}/127)")
    print(f"Tepi eksak: s = {s_edge:.4f} memberi 86/127 = {86/127:.4f}")
    print(f"Test n=115: cakupan s=1,85 -> {cov_te_185:.4f} ({int(round(cov_te_185*115))}/115)")
    print(f"Test n=115: cakupan s'={s_prime:.2f} -> {cov_te_prime:.4f} ({int(round(cov_te_prime*115))}/115)")

    # kurva cakupan validasi di sekitar s'
    curve = {}
    for ds_ in (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50):
        s_ = round(s_prime + ds_, 2)
        curve[f"{s_:.2f}"] = round(coverage(err_va, sd_va, s_), 4)

    mae_va = float(err_va.mean())
    mae_te = float(err_te.mean())

    out = {
        "metode": "s dipindai langkah 0,01 pada validasi Ca-P n=127 (cakupan +-1 sigma terdekat 0,68); "
                  "cakupan test dihitung dengan s yang sama (out-of-sample). Ensemble 5 seed, "
                  "prediksi test mereproduksi wp2_gnn_pretrained.json (max |d mean| %.2e)" % d_mean,
        "s_lama": 1.85,
        "s_prime": s_prime,
        "s_edge_exact": s_edge,
        "n_val": 127, "n_test": 115,
        "cakupan_validasi_s185": round(cov_va_185, 4),
        "cakupan_validasi_s_prime": round(cov_va_prime, 4),
        "cakupan_test_s185": round(cov_te_185, 4),
        "cakupan_test_s_prime": round(cov_te_prime, 4),
        "kurva_cakupan_validasi_di_sekitar_s_prime": curve,
        "mae_validasi": round(mae_va, 4),
        "mae_test": round(mae_te, 4),
        "sd_rerata_validasi": round(float(sd_va.mean()), 4),
        "sd_rerata_test": round(float(sd_te.mean()), 4),
    }
    json.dump(out, open(f"{BASE}/kalibrasi_s_validasi.json", "w"), indent=2)
    print("\nWrote kalibrasi_s_validasi.json (bagian 1/2: kalibrasi)")
    return out


if __name__ == "__main__":
    main()
