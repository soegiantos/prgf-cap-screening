# PRGF CaP Screening — Data and Code

Data and code for **"Domain- and Biology-Aware Screening of Calcium Phosphate
Bone-Scaffold Candidates with Graph Neural Networks and Density Functional
Theory Validation"** (submitted to *IEEE Access*).

Research Group Faculty Program (PRGF) 2026, Discovery Research scheme,
Universitas Airlangga, under Contract Number
10662/B/DST/UN3.FST/PT.01.03/2026.

## What this is

Finding new ion-substituted calcium phosphate compositions for bone scaffolds
by experiment or by single-structure computation is slow: one density
functional theory (DFT) label costs two to four days of continuous compute on
one machine. This repository documents the screening **mechanism** we tested —
a zoom-in ladder that protects expensive DFT labels by placing cheap,
knowledge-based gates before machine-learning ranking:

1. **Chemistry** — harvest and eligibility screening.
2. **Biology gate** — only literature-verified biocompatible ions
   (Zn, Sr, Mg, Si, Mn, F, Na, K, Cl).
3. **Crystal graph neural network (GNN) ensemble** — CGCNN-style surrogate,
   pretrained (Phase A, 31–60 epochs per member, mean 43) on 5,000 Materials
   Project phosphate structures and fine-tuned (Phase B, 11–16 epochs, mean
   13) on the Ca–P working dataset (815 entries).
4. **DFT validation** — Quantum ESPRESSO 7.5, PBE, MP2020 anion corrections.
5. **Concept analysis** — crystal-chemistry diagnosis of the failures.

On five DFT labels (one stable), the biology gate combined with the
GNN lower-confidence-bound rule (rule R5) selects the single stable candidate
and rejects all four unstable ones, and reduces the next DFT batch from ten
to three selected candidates. No new stable candidate is claimed — the
contribution is the tested mechanism itself. See the paper for the full
backtest (31 gate combinations, 24 orderings), the charge-balance filter,
and the limitations (n = 5 labels, one positive).

## Contents

```
prgf-cap-screening/
├── README.md
├── LICENSE            (MIT)
├── data/
│   ├── screening_r2.csv            Round-2 screening table (drafted candidates)
│   ├── screening_r2_nomad.csv      NOMAD subset of the round-2 screen
│   ├── screening_r2_tierAB.json    Tier A/B assignment of the screened pool
│   ├── consolidated_summary.json   Consolidated dataset summary (2,885 records / 950 compositions)
│   ├── wp2_dataset.json            Ca–P working dataset (WP2, 815 MP entries)
│   ├── wp2_gnn_pretrained.json     GNN ensemble test predictions + calibration
│   ├── dft_labels_r2_train.json    The five DFT labels used for backtesting
│   ├── aturan_seleksi_v2.json      Selection rules R0–R6b and their backtest
│   ├── charge_filter_r2.json       Charge-balance filter results
│   ├── kalibrasi_s_validasi.json   Uncertainty-calibration refit (s = 2.37)
│   ├── putaran2_draf30_v2.json     The 30 candidates drafted for DFT round 2
│   └── dft/
│       ├── r2_01_Ca3O8P2.out       DFT output — stable (beta-TCP itself)
│       ├── r2_16_Ca2O6PTb.out      DFT output — unstable
│       ├── r2_17_Ca2CeO8P.out      DFT output — unstable
│       ├── r2_18_CaFMgO4P.out      DFT output — verdict pending (mu_Mg)
│       ├── r2_19_Ca2H2O6P.out      DFT output — unstable
│       └── r2_24_Al2CaO6P.out      DFT output — unstable
└── code/
    ├── harvest_*.py, fetch_*.py    Harvesting from Materials Project / NOMAD / COD
    ├── consolidate.py              Deduplication and consolidation of harvested records
    ├── build_wp2_dataset.py        Build the Ca–P working dataset (WP2)
    ├── train_gnn_wp2.py            Pretrain (Phase A) + fine-tune (Phase B) the GNN ensemble
    ├── baseline_wp2.py             Composition-only random-forest baseline
    ├── aturan_seleksi_v2.py        Selection rules and backtesting
    ├── charge_filter.py            Charge-balance filter
    ├── kalibrasi_s_validasi.py     Uncertainty-calibration refit
    ├── finetune_r2.py              Fine-tuning on the new DFT labels (Phase C)
    ├── hitung_r6.py, dampak_s_prime.py   R6 soft-weight rule and s' analysis
    ├── prepare_qe_r2.py, prepare_qe_ref.py   Quantum ESPRESSO input preparation
    ├── merge_r2.py                 Merge round-2 results
    ├── fetch_hull_r2.py, fetch_hull_subsystems.py   Convex-hull queries (Materials Project)
    ├── analisis_dft_r1.py, analisis_out_of_domain.py, analisis_substitusi.py
    │                               Analysis of DFT labels and out-of-domain failures
    └── build_figures.py            Rebuild the paper figures
```

## Reproduction (summary)

1. **Harvest** (requires Materials Project and NOMAD API keys):
   `harvest_mp.py`, `harvest_mp_phosphate.py`, `harvest_nomad_structures.py`,
   `fetch_nomad_formulas.py`, `fetch_nomad_structures.py`, then
   `consolidate.py` and `build_wp2_dataset.py`.
2. **Train the GNN surrogate**: `train_gnn_wp2.py` (needs PyTorch and
   PyTorch Geometric; the pretrained predictions are already in
   `data/wp2_gnn_pretrained.json`).
3. **Screen and backtest**: `aturan_seleksi_v2.py`, `charge_filter.py`,
   `kalibrasi_s_validasi.py`.
4. **DFT validation**: `prepare_qe_r2.py` generates Quantum ESPRESSO inputs;
   the six completed outputs are in `data/dft/`. Formation energies are
   computed against Quantum ESPRESSO elemental references with MP2020 anion
   corrections, compared against the Materials Project convex hull.
5. **Figures**: `build_figures.py`.

Python 3.12 is used. The scripts are research code as run, not a packaged
library; paths inside the scripts refer to the original working directory
and may need adjusting.

## What is NOT included

The raw harvested records (155 MB, `data_raw/`) are not included in this
repository. The processed dataset (`data/wp2_dataset.json`,
`data/consolidated_summary.json`) and the screening tables above are the
complete basis for the results reported in the paper. The raw records can
be re-harvested with the scripts in `code/` from the public sources
(Materials Project, AFLOW, NOMAD, COD, PubChem).

## License

MIT License — see `LICENSE`. You are free to use, modify, and redistribute
this data and code, with attribution.

## Citation

If you use this data or code, please cite the paper (IEEE Access, 2026):

> Soelistiono, S., Trilaksana, H., Yhuwana, Y. G. Y., & Pranggono, B.
> "Domain- and Biology-Aware Screening of Calcium Phosphate Bone-Scaffold
> Candidates with Graph Neural Networks and Density Functional Theory
> Validation." *IEEE Access*, 2026.
