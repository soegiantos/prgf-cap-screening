#!/usr/bin/env python3
"""WP2 surrogate GNN — PRGF baru (venv python: torch + pymatgen).
Arsitektur: crystal graph convolution gaya CGCNN (Xie & Grossman, PRL 2018)
yang diimplementasi ringkas di sini (bukan paket cgcnn, supaya auditabel):
  - tiap atom -> embedding unsur (dim 64, dipelajari)
  - tetangga dalam radius 8 A (maks 12 tetangga), PBC lewat gambar periodik
  - 3 lapis konvolusi graf dengan gating (sigmoid * softplus)
  - pooling rata-rata atas atom -> MLP -> energi formasi per atom
  - ensemble 5 anggota (seed berbeda) -> prediksi = rerata, ketidakpastian = sd
Data: wp2_dataset.json (MP 815; target formation_energy_per_atom).
Split: train/val/test grouped by rumus (573/127/115).
Fitur tepi: jarak Gaussian (40 basis, 0-8 A).
Latih: Adam lr 1e-3, batch 32 graf, 120 epoch, early stop di val (patience 20).
Metrik: MAE/RMSE/R2 di test; bandingkan vs baseline (RF test R2=0.725).
Juga laporkan kalibrasi ketidakpastian ensemble (apakah sd memprediksi |error|).
"""
import json, math, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Device: MPS di Mac (bila tersedia), CPU bila tidak. Bisa dipaksa lewat env WP2_DEVICE=cpu|mps|cuda.
DEVICE = torch.device(os.environ.get("WP2_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Device latih: {DEVICE}", flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ds = json.load(open(f"{BASE}/wp2_dataset.json"))
mp = ds["mp_entries"]

ELEMENTS = sorted({el for e in mp for el in e["counts"]})
ELIDX = {el: i + 1 for i, el in enumerate(ELEMENTS)}  # 0 = padding
from pymatgen.core import Element
ZMAP = {el: Element(el).Z for el in ELEMENTS}
MAX_NBR, RADIUS = 12, 8.0
BATCH = 8  # kecil: VM hanya ~1,5 CPU dan RAM ketat; batch 32 membuat tensor padding raksasa -> swap

def build_graph(e):
    lat = np.array(e["lattice"], float)
    frac = np.array(e["frac_coords"], float)
    els = np.array([ELIDX[x] for x in e["elements"]], dtype=np.int64)
    n = len(els)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    nbr_idx = np.zeros((n, MAX_NBR), dtype=np.int64)
    nbr_dist = np.zeros((n, MAX_NBR), dtype=np.float32)
    for a0 in range(0, n, 64):  # blok atom demi hemat memori
        a1 = min(a0 + 64, n)
        # diff[a, b, s, :] = frac[b] + shift[s] - frac[a]
        diff = frac[None, :, None, :] + shifts[None, None, :, :] - frac[a0:a1, None, None, :]
        cart = diff @ lat
        dist = np.linalg.norm(cart, axis=-1)  # (block, n, 27)
        blk = np.arange(a1 - a0)
        dist[blk, np.arange(a0, a1), 13] = np.inf  # hanya gambar (0,0,0) diri sendiri
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

print("Membangun graf kristal ...", flush=True)
graphs = [build_graph(e) for e in mp]
print(f"Graf: {len(graphs)}; unsur: {len(ELEMENTS)}; max situs: {max(len(g['atom']) for g in graphs)}", flush=True)

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
            # keluaran 2*emb: paruh pertama gerbang (sigmoid), paruh kedua pesan (softplus)
            self.conv.append(nn.Linear(2 * emb + n_gauss, 2 * emb))
        self.out1 = nn.Linear(emb, 64)
        self.out2 = nn.Linear(64, 1)
        self.act = nn.Softplus()
    def forward(self, atom, nbr_idx, nbr_dist, mask):
        h = self.emb(atom)  # B,N,emb
        # fitur tepi Gaussian
        g = torch.exp(-((nbr_dist.unsqueeze(-1) - self.centers.to(nbr_dist.device)) ** 2) / 0.25)  # B,N,K,G
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

def train_one(seed, epochs=60):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr = [g for g in graphs if g["split"] == "train"]
    va = [g for g in graphs if g["split"] == "val"]
    te = [g for g in graphs if g["split"] == "test"]
    model = GNN(len(ELEMENTS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    # batching berkelompok menurut ukuran (nsites) supaya padding tidak boros:
    # urutkan menurut nsites, potong jadi batch 8, acak urutan batch tiap epoch
    order = sorted(range(len(tr)), key=lambda i: len(tr[i]["atom"]))
    batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        random.shuffle(batches)
        tot = 0.0
        for bidx in batches:
            batch = [tr[i] for i in bidx]
            atom, ni, nd, mk, y = [t.to(DEVICE) for t in collate(batch)]
            opt.zero_grad()
            loss = lossf(model(atom, ni, nd, mk), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            vl = DataLoader(DS(va), batch_size=64, collate_fn=collate)
            errs = []
            for atom, ni, nd, mk, y in vl:
                atom, ni, nd, mk, y = atom.to(DEVICE), ni.to(DEVICE), nd.to(DEVICE), mk.to(DEVICE), y.to(DEVICE)
                errs.append((model(atom, ni, nd, mk) - y).cpu().numpy())
            errs = np.concatenate(errs)
            vmse = float((errs ** 2).mean())
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  seed {seed} ep {ep+1}: train_MSE={tot/len(tr):.4f} val_RMSE={math.sqrt(vmse):.4f}", flush=True)
        if vmse < best - 1e-5:
            best, bad = vmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        for name, subset in [("val", va), ("test", te)]:
            dl = DataLoader(DS(subset), batch_size=64, collate_fn=collate)
            ps, ys = [], []
            for atom, ni, nd, mk, y in dl:
                atom, ni, nd, mk, y = atom.to(DEVICE), ni.to(DEVICE), nd.to(DEVICE), mk.to(DEVICE), y.to(DEVICE)
                ps.append(model(atom, ni, nd, mk).cpu().numpy()); ys.append(y.cpu().numpy())
            out[name] = (np.concatenate(ps), np.concatenate(ys))
    print(f"  seed {seed}: epochs={ep+1} val_RMSE={math.sqrt(best):.4f} "
          f"test_MAE={np.abs(out['test'][0]-out['test'][1]).mean():.4f} "
          f"test_R2={1 - ((out['test'][0]-out['test'][1])**2).sum()/((out['test'][1]-out['test'][1].mean())**2).sum():.3f}", flush=True)
    return out["test"][0], out["test"][1]

torch.set_num_threads(int(os.environ.get("WP2_THREADS", "2")))
PARTIAL = f"{BASE}/wp2_gnn_partial.json"
print("Melatih ensemble 5 anggota ...", flush=True)
preds, ytrue = [], None
done = {}
if os.path.exists(PARTIAL):
    try:
        done = json.load(open(PARTIAL))
        print(f"Melanjutkan: {len(done)} anggota ensemble sudah selesai sebelumnya.", flush=True)
    except Exception:
        done = {}
for s in range(5):
    key = f"seed_{100 + s}"
    if key in done:
        p = np.array(done[key]["pred"])
        ytrue = np.array(done[key]["ytrue"])
        preds.append(p)
        print(f"  seed {100 + s}: sudah ada (test_MAE={np.abs(p - ytrue).mean():.4f})", flush=True)
        continue
    p, ytrue = train_one(100 + s)
    preds.append(p)
    done[key] = {"pred": [float(x) for x in p], "ytrue": [float(x) for x in ytrue]}
    json.dump(done, open(PARTIAL, "w"))  # simpan tiap anggota selesai -> tahan restart VM
    print(f"  seed {100 + s}: tersimpan ke {PARTIAL}", flush=True)
P = np.stack(preds)
mean, sd = P.mean(axis=0), P.std(axis=0)
err = np.abs(mean - ytrue)
print(f"\nENSEMBLE (5) TEST: MAE={err.mean():.4f} RMSE={np.sqrt(((mean-ytrue)**2).mean()):.4f} "
      f"R2={1 - ((mean-ytrue)**2).sum()/((ytrue-ytrue.mean())**2).sum():.3f}")
# kalibrasi ketidakpastian: korelasi sd vs |error|; cakupan interval +-1 sd / +-2 sd
from scipy.stats import pearsonr
print(f"Kalibrasi: corr(sd, |error|)={pearsonr(sd, err)[0]:.3f}; "
      f"cakupan +-1sd={np.mean(err <= sd):.2f}, +-2sd={np.mean(err <= 2*sd):.2f}")
print(f"Rerata sd ensemble={sd.mean():.4f} eV/atom (target ketidakpastian < 0.10 untuk seleksi aktif)")

json.dump({"test": {"mae": float(err.mean()), "rmse": float(np.sqrt(((mean-ytrue)**2).mean())),
                   "r2": float(1 - ((mean-ytrue)**2).sum()/((ytrue-ytrue.mean())**2).sum())},
           "calibration": {"corr_sd_abs_err": float(pearsonr(sd, err)[0]),
                           "coverage_1sd": float(np.mean(err <= sd)),
                           "coverage_2sd": float(np.mean(err <= 2*sd)),
                           "mean_sd": float(sd.mean())}},
          open(f"{BASE}/wp2_gnn.json", "w"), indent=2)
print("Wrote wp2_gnn.json")
