# Hyperalignment Reveals Shared Information in Idiosyncratic Fine-Grained Movie fMRI Patterns in Macaques

Code release accompanying the paper:

> Zhang, Y., Wang, H., Zhu, Q., Li, X., Han, J., Jiahui, G., Shim, W.M., Kim, R., Gobbini, M.I.,
> Heo, K., Sepe, A., Haxby, J.V., Feilong, M., Vanduffel, W. *Hyperalignment Reveals Shared
> Information in Idiosyncratic Fine-Grained Movie fMRI Patterns in Macaques.*

This repository contains the analysis code used to produce every main and supplementary figure
in the paper: response hyperalignment (RHA) of movie-viewing fMRI data in macaques and humans,
inter-subject correlation, between-subject movie time-point classification, prediction of
individual functional localizer topographies, and the movie-repetition control analysis.

## Structure

Each numbered folder pairs the heavy computation (`compute_*.py` / `predict_*.py` /
`pairwise_*.py`, meant to be run on a cluster) with a lightweight `plot_*.ipynb` notebook that
loads the resulting arrays and reproduces the corresponding figure.

- **`01_hyperalignment/`** — Builds the group common-space templates and per-subject
  response-hyperalignment transforms (searchlight-based, radius = 20 mm) for both species.
  - `compute_templates_monkey.py`, `compute_templates_human.py`
  - `hyperalign_to_templates_monkey.py`, `hyperalign_to_templates_human.py`

- **`02_isc/`** — Inter-subject correlation of movie-evoked responses, anatomically-aligned vs.
  hyperaligned (**Fig. 1**, **Supp. Fig. S2**).
  - `compute_isc_monkey.py`, `compute_isc_human.py`
  - `plot_isc.ipynb`

- **`03_movie_timepoint_classification/`** — Between-subject decoding of 2 s movie time-points,
  whole-brain and searchlight-based (**Fig. 2**, **Supp. Fig. S3**).
  - `classify_timepoints_wholebrain_monkey.py`, `classify_timepoints_wholebrain_human.py`
  - `classify_timepoints_searchlight_monkey.py`, `classify_timepoints_searchlight_human.py`
  - `plot_timepoint_classification.ipynb`

- **`04_functional_map_prediction/`** — Pairwise RHA transforms used to predict one subject's
  face/body-selective localizer topography from every other subject's data (**Fig. 3**,
  **Supp. Figs. S1, S4–S6**).
  - `pairwise_hyperalignment_monkey.py`, `pairwise_hyperalignment_human.py`
  - `predict_localizer_maps_monkey.py`, `predict_localizer_maps_human.py`
  - `plot_localizer_maps_monkey.ipynb`, `plot_localizer_maps_human.ipynb`

- **`05_movie_repetition_analysis/`** — Effect of the number of movie repetitions used to build
  the RHA transform vs. used to evaluate it on held-out ISC (**Supp. Fig. S7**, monkey only).
  Repetitions were quality-controlled by Cronbach's alpha before averaging.
  - `compute_templates_diff_rep.py`, `compute_transforms_diff_rep.py`
  - `compute_isc_diff_rep.py`, `assemble_isc_grid_diff_rep.py`
  - `plot_isc_diff_rep.ipynb`

## Requirements

- Python 3, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `nibabel`, `joblib`, `Pillow`
- [`neuroboros`](https://github.com/neuroboros/neuroboros) — cortical surface I/O, searchlights,
  ISC, and plotting (`nb.plot_mebrains` for macaque, `nb.plot` for human)
- The `hyperalignment` package providing `searchlight_procrustes`, `searchlight_ridge`,
  `searchlight_template`, and `searchlight_weights`

## Data

The `compute_*`/`predict_*`/`pairwise_*` scripts read from and write to fixed cluster paths
(under `/dartfs/rc/lab/H/HaxbyLab/...`) that are not included in this repository. To rerun the
pipeline on your own data, point `DATA_ROOT`, `MAPS_ROOT`, and `root`/`OUT_ROOT` at the top of
each script to your own copies of the macaque (`Monkey Kingdom`) and human movie-viewing and
localizer datasets, resampled to the `mkavg-ico32` and `onavg-ico32` cortical surface templates
respectively. The `plot_*.ipynb` notebooks only need the `.npy`/`.npz` outputs already produced
by the corresponding compute scripts.
