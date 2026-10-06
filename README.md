# Hybrid tuberculosis lesion model: publication code

This folder packages the Python code accompanying **“A mechanistic spatio-temporal PKPD model to compare generations of diarylquinolines: Bedaquiline versus TBAJ-587 in a rabbit model of active TB”** (Bailo et al., manuscript supplied with this project). The model combines a diffusion–reaction description of drug and oxygen distribution with a stochastic, cell-based simulation of infection and treatment. The paper's reference workflows use 20 µm agent cells, a 0.01-day agent step, a 0.1-day diffusion step, 84 untreated days, 180 treatment days, and follow-up to day 500 for the circular ABM.

This release copy has English comments, a command entry point, explicit dependencies, and paths resolved from this folder.

## Setup

Use Python 3.12. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

The 2D simulations and large replicate sets may take substantial time and produce many files. Start with one run per drug; increase `--simulations` when reproducing ensemble analyses.

## Entry point

```bash
# Fitted diffusion coefficient D(distance), corresponding to Figure 3a-b
python run.py diffusion-profile --drug BDQ
python run.py diffusion-profile --drug TBAJ587

# 30-day radial fit with embedded LCM observations and 500 PK samples (Figure 3c-d)
python run.py radial-drug --drug BDQ --days 30 --observations --pk-variability --simulations 500
python run.py radial-drug --drug TBAJ587 --days 30 --observations --pk-variability --simulations 500

# 180 treatment days plus 170 washout days in a 3.4 mm radial domain (Figure 3e-f / Table 3)
python run.py radial-drug --drug BDQ --diameter 3400 --donuts 5
python run.py radial-drug --drug TBAJ587 --diameter 3400 --donuts 5

# One 3.4 mm circular-domain agent run per drug (Figures 4-6)
python run.py circular-abm --drug BDQ --seed 1234567
python run.py circular-abm --drug TBAJ587 --diameter 3400 --donuts 5 --seed 1234567

# Irregular-domain drug calculation from a supplied boundary (Table 4 component)
python run.py irregular-drug --drug BDQ --geometry data/geometry/boundary.csv
```

The commands write timestamped runs under `results/`. `--output-dir` accepts another path relative to this release folder. `--dry-run` checks the chosen script and inputs without executing the model. `--pk-variability --simulations N` samples the PK parameters in the radial and irregular drug workflows. `--simulations N` repeats the circular ABM with consecutive seeds; each replicate gets its own folder. Output includes CSV time series, parameter summaries, plots, and, for circular ABM runs, snapshot PNGs and GIFs. The circular script saves snapshots at days 5, 25, 50, 84, 100, 200, 250, 300, 350, and the final day.

For an irregular boundary, place a CSV in `data/geometry/` with columns `time id,points`; `points` is a Python-style list of grid coordinates, for example `"[[10,10],[10,11],...]"`. The boundary must be a valid closed outline, ordered around the region, and the grid indices must match the chosen mesh. Omitting `--geometry` opens the interactive lasso workflow. An optional `--image data/geometry/image.png` puts a background beneath the selection. The original manuscript's LCM-derived outlines and image are **not** included here, so the irregular-domain examples require the user to provide or draw a boundary.

## Figure and table coverage

| Manuscript item | Code here | Reproduction scope |
| --- | --- | --- |
| Fig. 1, LCM image and donut diagram | — | Source microscopy/diagram is not supplied. |
| Fig. 2, agent-rule diagram | `src/functions.py`, `src/stochastic.py` | Rules are implemented; the published schematic is not generated. |
| Fig. 3a-b, fitted spatial diffusion function | `diffusion-profile` | Generates the fitted curves from Table 2 parameters. |
| Fig. 3c-d, 30-day drug fit | `radial-drug --days 30 --observations --pk-variability --simulations 500` | Generates concentration curves and overlays the LCM values embedded in the source. Plot layout and random bands may differ from the publication. |
| Fig. 3e-f and Table 3, 180-day treatment and washout | `radial-drug --diameter 3400 --donuts 5` | Generates radial concentration and threshold outputs for each drug. TBAJ-587's calibration defaults to four donuts; the five-donut option matches the paper's Table 3 setup. |
| Figs. 4-5, circular ABM snapshots | `circular-abm` | Generates drug, oxygen, cell, and intracellular-Mtb snapshots for individual stochastic runs. |
| Fig. 6, agent and oxygen time courses | `circular-abm --simulations N` | Produces per-run trajectories and quantity plots. The paper's 100-run overlay is not assembled automatically. |
| Figs. 7-8 and Table 4, irregular domains | `irregular-drug` | The irregular drug-diffusion/threshold component can be computed for a supplied boundary. The irregular-domain **ABM** used for published snapshots and eradication probabilities is absent from the supplied source files. |
| Table 5, eradication probabilities | `circular-abm --simulations N` | Replicates can be run, but there is no validated aggregation and confidence-interval workflow in these files. The paper reports 1,000 circular and 100 irregular ABM runs per drug. |

The manuscript's exact random seeds, original irregular boundaries, LCM microscopy, and complete plotting/aggregation scripts were not present in the source folder. Consequently these commands reproduce model outputs and figure components, but cannot be claimed to regenerate every published panel or reported statistic exactly.

## Layout

```text
publication_release/
├── run.py                     # command entry point
├── requirements.txt
├── src/
│   ├── functions.py           # shared solvers, PK and agent rules
│   ├── simulation_1d.py       # radial diffusion and PK variability
│   ├── stochastic.py          # circular-domain hybrid ABM
│   ├── diffusion_manual_domain.py # user-defined 2D drug domain
│   ├── plot_diffusion_profile.py
│   ├── release_config.py      # project-relative paths
│   └── SelectFromCollection.py # interactive lasso selector
├── data/geometry/             # user-supplied or drawn boundaries
└── results/                   # generated outputs
```

The drug parameters and embedded calibration measurements are in the two drug scripts. The original workflow uses stochastic initialization; use `--seed` for repeatable circular ABM runs. Runs are generated from source parameters and should be checked against the final published manuscript before public release.

Copyright (c) 2026, Fondazione The Microsoft Research - University of Trento Centre for Computational and Systems Biology(COSBI)
All rights reserved.