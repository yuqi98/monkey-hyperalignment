import os
import sys
from glob import glob

import neuroboros as nb
import numpy as np
from scipy.stats import zscore

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkey/nb/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
OUT_ROOT = f"{root}/rep_analysis"
TRANSFORM_ROOT = f"{OUT_ROOT}/transforms"
ISC_ROOT = f"{OUT_ROOT}/isc"
MAX_REPS = 7
ALIGNMENTS = ["procr", "ridge"]


def get_response_data_by_rep(sid, runs, mask, n_reps):
    dms = []
    for run in runs:
        dm_l = np.load(f"{DATA_ROOT}/{sid}/clip{run}_lh.npy")
        dm_r = np.load(f"{DATA_ROOT}/{sid}/clip{run}_rh.npy")
        picked = np.load(f"{DATA_ROOT}/{sid}/clip{run}_mask.npy")
        good = np.where(picked)[0]
        assert len(good) >= n_reps, (
            f"{sid} clip{run} has only {len(good)} usable repetitions, need {n_reps}"
        )
        sel = good[:n_reps]

        dm_l_avg = np.mean(dm_l[sel], axis=0)
        dm_r_avg = np.mean(dm_r[sel], axis=0)

        if dm_l_avg.shape[0] == 900:
            num_vertices_l = dm_l_avg.shape[1]
            num_vertices_r = dm_r_avg.shape[1]
            dm_l_avg = dm_l_avg.reshape(450, 2, num_vertices_l).mean(axis=1)
            dm_r_avg = dm_r_avg.reshape(450, 2, num_vertices_r).mean(axis=1)

        dm_full = np.hstack([dm_l_avg, dm_r_avg])
        dms.append(dm_full[:, mask])

    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


def compute_isc(all_data, sids, align=None, transform_rep=None):
    dms = []
    for sid in sids:
        dm = all_data[sid]
        if align is not None:
            transform = np.load(
                f"{TRANSFORM_ROOT}/monkey_{sid}_3_clips_all_monkey_{align}"
                f"_with_mask_unweighted_rep_{transform_rep}.npy",
                allow_pickle=True,
            )
            dm = np.nan_to_num(zscore(dm @ transform[()], axis=0))
        dms.append(dm)
    dms = np.stack(dms, axis=0)
    return nb.isc(dms, pairwise=False, metric="correlation")


if __name__ == "__main__":
    os.makedirs(ISC_ROOT, exist_ok=True)

    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print("Subjects found:", sids)
    assert len(sids) == 12

    mask = nb.mask("lr", "mkavg-ico32")

    combos = [("anatomical", None, d) for d in range(1, MAX_REPS + 1)]
    combos += [
        (align, r, d)
        for align in ALIGNMENTS
        for r in range(1, MAX_REPS + 1)
        for d in range(1, MAX_REPS + 1)
    ]

    arg = int(sys.argv[1])
    align, transform_rep, data_rep = combos[arg]
    print(f"align={align} transform_rep={transform_rep} data_rep={data_rep}")

    if align == "anatomical":
        out_fn = f"{ISC_ROOT}/anatomical_with_mask_data_rep_{data_rep}.npy"
    else:
        out_fn = (
            f"{ISC_ROOT}/hyperaligned_3_clips_all_monkey_{align}_with_mask_unweighted"
            f"_transform_rep_{transform_rep}_data_rep_{data_rep}.npy"
        )
    if os.path.exists(out_fn):
        print(f"{out_fn} already exists, skipping")
        sys.exit(0)

    all_data = {}
    for sid in sids:
        dm = get_response_data_by_rep(sid, [4, 5], mask, n_reps=data_rep)
        assert dm.shape == (900, 19863)
        all_data[sid] = dm

    print(f"Saving ISC to: {out_fn}")
    if align == "anatomical":
        nb.record(out_fn, compute_isc)(all_data, sids)
    else:
        nb.record(out_fn, compute_isc)(all_data, sids, align=align, transform_rep=transform_rep)
