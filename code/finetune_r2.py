#!/usr/bin/env python3
"""Fine-tune Fase C GNN — PRGF baru (dijalankan di Mac bila bobot ada, atau VM CPU).

Latar: surrogate WP2 (pretrain 5.000 fosfat MP + fine-tune Ca-P, 5 seed,
test R2 0,950 / MAE 0,087) sistematis melebih-lebihkan kestabilan struktur
NOMAD eksotis putaran 2: meleset DFT-GNN +0,17 (r2_24), +0,20 (r2_16),
+0,79 (r2_19), +1,09 (r2_17) eV/atom. Analisis out-of-domain
(dft_r2/analisis_out_of_domain.json): sd ensemble baris NOMAD rata-rata
0,114 vs MP 0,052 — sd TIDAK menandai kesalahan (r2_17: sd 0,090 tapi
meleset 1,09 ≈ 12x sd). Ensemble terlalu percaya diri di luar domain.

Fase C: lanjutkan latih tiap anggota ensemble (bobot gnn_seed_{100+s}.pt,
fase "finetuned") dengan 4 label QE baru putaran 2 (r2_16, r2_17, r2_19,
r2_24; E_form TERKOREKSI MP2020 — skala sama dengan target MP).
  - Validasi jujur: leave-one-out (4 lipatan) — tiap lipatan fine-tune di
    3 label lain, uji di label yang ditahan. Dengan 4 label saja, LOO
    adalah satu-satunya ukuran yang tidak menipu diri sendiri.
  - Anti-lupa (catastrophic forgetting): tiap epoch mencampur label baru
    (diulang 8x) dengan 64 graf replay acak dari split train Ca-P.
  - LR 1e-4, 10 epoch tetap (tanpa early stop — terlalu sedikit sampel).
  - Model akhir: fine-tune di ke-4 label, simpan gnn_seed_{100+s}_r2.pt.
  - Cek lupa: MAE di 50 graf test Ca-P acak, sebelum vs sesudah Fase C.
Keluaran: wp2_gnn_r2_finetuned.json (hasil LOO + prediksi baru ke-4 label
+ cek lupa). Bobot baru TIDAK menimpa bobot lama.
Jalankan: .venv-wp2/bin/python finetune_r2.py  (WP2_DEVICE=cpu|mps)
"""
import json, math, random, os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--freeze-conv", action="store_true",
                help="bekukan embedding + konvolusi, latih hanya kepala output (out1/out2)")
ap.add_argument("--tag", default="",
                help="akhiran berkas keluaran, mis. -head (default: tanpa akhiran)")
ap.add_argument("--replay-n", type=int, default=64,
                help="jumlah graf replay Ca-P train (default 64)")
args = ap.parse_args()
TAG = args.tag
print(f"Varian: {'kepala-output saja (conv beku)' if args.freeze_conv else 'semua parameter'}"
      f"{f' | tag {TAG}' if TAG else ''}", flush=True)

DEVICE = torch.device(os.environ.get("WP2_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Device latih: {DEVICE}", flush=True)
BASE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("WP2_THREADS", "4")))

MAX_NBR, RADIUS = 12, 8.0
BATCH = 8
LR = 1e-4
EPOCHS = 10
REPEAT_NEW = 8      # label baru diulang 8x tiap epoch (data sangat sedikit)
REPLAY_N = args.replay_n  # graf replay Ca-P train tiap lipatan (anti-lupa)
SEEDS = [100, 101, 102, 103, 104]

# ---------- graf & model (salinan persis pretrain_gnn.py) ----------
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
            "y": float(e["formation_e"]), "mid": e["material_id"]}

class DS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

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
        atom[i, :n] = b["atom"]; nbr_idx[i, :n] = b["nbr_idx"]
        nbr_dist[i, :n] = b["nbr_dist"]; mask[i, :n] = 1.0; y[i] = b["y"]
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
            atom, ni, nd, mk, y = atom.to(DEVICE), ni.to(DEVICE), nd.to(DEVICE), mk.to(DEVICE), y.to(DEVICE)
            ps.append(model(atom, ni, nd, mk).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)

def train_epochs(model, items, epochs, lr, seed):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["atom"]))
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    for ep in range(epochs):
        model.train()
        random.shuffle(batches)
        tot = 0.0
        for bidx in batches:
            batch = [items[i] for i in bidx]
            atom, ni, nd, mk, y = [t.to(DEVICE) for t in collate(batch)]
            opt.zero_grad()
            loss = lossf(model(atom, ni, nd, mk), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(batch)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    seed {seed} ep {ep+1}: train_MSE={tot/len(items):.4f}", flush=True)

def bekukan_kecuali_kepala(model):
    """Varian head-only: bekukan embedding + konvolusi, latih hanya out1/out2."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("out")

# ---------- muat bobot & data ----------
ckpt_paths = [f"{BASE}/gnn_seed_{s}.pt" for s in SEEDS]
missing = [p for p in ckpt_paths if not os.path.exists(p)]
if missing:
    print("BOBOT TIDAK DITEMUKAN — fine-tune Fase C tidak bisa jalan:")
    for p in missing:
        print("  hilang:", p)
    print("Bobot gnn_seed_100..104.pt ada di Mac ~/prgf-wp2/ (Mac sedang tidak bisa dihubungi).")
    print("Salin kelima berkas .pt itu ke folder skrip ini, lalu jalankan lagi.")
    raise SystemExit(1)

ckpts = [torch.load(p, map_location="cpu") for p in ckpt_paths]
ELEMENTS = ckpts[0]["elements"]
ELIDX = {el: i + 1 for i, el in enumerate(ELEMENTS)}
print(f"Bobot dimuat: 5 seed, fase={ckpts[0].get('phase')}, unsur={len(ELEMENTS)}", flush=True)

labels = json.load(open(f"{BASE}/dft_labels_r2_train.json"))["records"]
new = [r for r in labels if not r.get("duplikat_mp")]  # r2_01 duplikat MP -> bukan label baru
print(f"Label QE baru: {len(new)} ({', '.join(r['job'] for r in new)})", flush=True)
unknown = {el for r in new for el in r["elements"] if el not in ELIDX}
if unknown:
    print(f"UNSUR TIDAK DIKENAL MODEL: {sorted(unknown)} — fine-tune dibatalkan.")
    raise SystemExit(1)
new_graphs = [build_graph(r, ELIDX) for r in new]

cap = json.load(open(f"{BASE}/wp2_dataset.json"))["mp_entries"]
cap_tr = [e for e in cap if e["split"] == "train"]
cap_te = [e for e in cap if e["split"] == "test"]
rng = random.Random(7)
replay_pool = rng.sample(cap_tr, REPLAY_N)
replay_graphs = [build_graph(e, ELIDX) for e in replay_pool]
forget_sample = rng.sample(cap_te, 50)
forget_graphs = [build_graph(e, ELIDX) for e in forget_sample]
print(f"Replay: {len(replay_graphs)} graf Ca-P train; cek lupa: {len(forget_graphs)} graf Ca-P test", flush=True)

# prediksi AWAL (bobot lama) di label baru + sampel lupa
def ensemble_predict(models, subset):
    ps = [predict(m, subset)[0] for m in models]
    P = np.stack(ps)
    return P.mean(axis=0), P.std(axis=0)

models0 = []
for s, ck in zip(SEEDS, ckpts):
    m = GNN(len(ELEMENTS)).to(DEVICE)
    m.load_state_dict({k: v.to(DEVICE) for k, v in ck["state_dict"].items()})
    models0.append(m)
y_new = np.array([g["y"] for g in new_graphs])
mean0, sd0 = ensemble_predict(models0, new_graphs)
p_forget0, y_forget = predict(models0[0], forget_graphs)  # cek lupa per-seed dihitung di akhir
forget0 = [float(np.abs(predict(m, forget_graphs)[0] - y_forget).mean()) for m in models0]
print(f"MAE awal di 4 label baru: {np.abs(mean0 - y_new).mean():.4f} "
      f"(meleset per label: {', '.join(f'{a:.2f}' for a in np.abs(mean0 - y_new))})", flush=True)
print(f"MAE awal di 50 test Ca-P (cek lupa): {np.mean(forget0):.4f}", flush=True)

# ---------- LOO-CV ----------
loo_rows = []
for hold in range(len(new_graphs)):
    tr_idx = [i for i in range(len(new_graphs)) if i != hold]
    tr_items = [new_graphs[i] for _ in range(REPEAT_NEW) for i in tr_idx] + replay_graphs
    held = [new_graphs[hold]]
    preds = []
    for s, ck in zip(SEEDS, ckpts):
        torch.manual_seed(s); random.seed(s); np.random.seed(s)
        m = GNN(len(ELEMENTS)).to(DEVICE)
        m.load_state_dict({k: v.to(DEVICE) for k, v in ck["state_dict"].items()})
        if args.freeze_conv:
            bekukan_kecuali_kepala(m)
        train_epochs(m, tr_items, EPOCHS, LR, s)
        p, _ = predict(m, held)
        preds.append(float(p[0]))
    P = np.array(preds)
    loo_rows.append({"job": new[hold]["job"], "ytrue": float(y_new[hold]),
                     "pred_mean_sebelum": float(mean0[hold]), "sd_sebelum": float(sd0[hold]),
                     "pred_mean_sesudah": float(P.mean()), "sd_sesudah": float(P.std()),
                     "meleset_sebelum": float(abs(mean0[hold] - y_new[hold])),
                     "meleset_sesudah": float(abs(P.mean() - y_new[hold]))})
    print(f"LOO {new[hold]['job']}: DFT {y_new[hold]:.4f} | sebelum {mean0[hold]:.4f} "
          f"(meleset {abs(mean0[hold]-y_new[hold]):.4f}) -> sesudah {P.mean():.4f} "
          f"(meleset {abs(P.mean()-y_new[hold]):.4f})", flush=True)

mae_before = float(np.mean([r["meleset_sebelum"] for r in loo_rows]))
mae_after = float(np.mean([r["meleset_sesudah"] for r in loo_rows]))
print(f"\nLOO MAE: sebelum {mae_before:.4f} -> sesudah {mae_after:.4f}", flush=True)

# ---------- model akhir: fine-tune di ke-4 label ----------
final_items = [g for _ in range(REPEAT_NEW) for g in new_graphs] + replay_graphs
final_models = []
for s, ck in zip(SEEDS, ckpts):
    torch.manual_seed(s); random.seed(s); np.random.seed(s)
    m = GNN(len(ELEMENTS)).to(DEVICE)
    m.load_state_dict({k: v.to(DEVICE) for k, v in ck["state_dict"].items()})
    if args.freeze_conv:
        bekukan_kecuali_kepala(m)
    print(f"Model akhir seed {s} ...", flush=True)
    train_epochs(m, final_items, EPOCHS, LR, s)
    torch.save({"state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
                "elements": ELEMENTS, "seed": s, "phase": "finetuned-r2"},
               f"{BASE}/gnn_seed_{s}_r2{TAG}.pt")
    final_models.append(m)
meanF, sdF = ensemble_predict(final_models, new_graphs)
forgetF = [float(np.abs(predict(m, forget_graphs)[0] - y_forget).mean()) for m in final_models]
print(f"Prediksi akhir ke-4 label: {', '.join(f'{a:.4f}' for a in meanF)} "
      f"(DFT: {', '.join(f'{a:.4f}' for a in y_new)})", flush=True)
print(f"Cek lupa 50 test Ca-P: MAE sebelum {np.mean(forget0):.4f} -> sesudah {np.mean(forgetF):.4f}", flush=True)

out = {"loo": loo_rows,
       "loo_mae_sebelum": mae_before, "loo_mae_sesudah": mae_after,
       "prediksi_akhir_label_baru": [
           {"job": r["job"], "ytrue": float(y), "pred_mean": float(pm), "pred_sd": float(ps)}
           for r, y, pm, ps in zip(new, y_new, meanF, sdF)],
       "cek_lupa_50_test_cap": {"mae_sebelum": float(np.mean(forget0)),
                                "mae_sesudah": float(np.mean(forgetF))},
       "meta": {"lr": LR, "epochs": EPOCHS, "repeat_new": REPEAT_NEW,
                "replay_n": REPLAY_N, "seeds": SEEDS, "freeze_conv": args.freeze_conv,
                "label_baru": [r["job"] for r in new],
                "bobot_baru": [f"gnn_seed_{s}_r2{TAG}.pt" for s in SEEDS],
                "catatan": "LOO: tiap lipatan fine-tune di 3 label + 64 replay Ca-P train."}}
json.dump(out, open(f"{BASE}/wp2_gnn_r2_finetuned{TAG}.json", "w"), indent=2)
print(f"Wrote wp2_gnn_r2_finetuned{TAG}.json + gnn_seed_*_r2{TAG}.pt (bobot lama tidak ditimpa)")
