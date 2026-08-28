import os
import sys
from glob import glob

import neuroboros as nb
import numpy as np
from scipy.stats import zscore
from hyperalignment import searchlight_template

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkey/nb/mkavg-ico32"
OUT_ROOT = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/rep_analysis"
TEMPLATE_ROOT = f"{OUT_ROOT}/templates"
radius = 20
MAX_REPS = 7


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


if __name__ == "__main__":
    os.makedirs(TEMPLATE_ROOT, exist_ok=True)

    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print("Subjects found:", sids)
    assert len(sids) == 12

    mask = nb.mask("lr", "mkavg-ico32")
    sls, dists = nb.sls("lr", radius, space="mkavg-ico32", return_dists=True, mask=True)
    assert len(sls) == 19863

    arg = int(sys.argv[1])
    rep = arg + 1
    assert 1 <= rep <= MAX_REPS

    out_fn = f"{TEMPLATE_ROOT}/monkey_3_clips_with_mask_rep_{rep}.npy"
    if os.path.exists(out_fn):
        print(f"{out_fn} already exists, skipping")
        sys.exit(0)

    print(f"Building template using {rep} repetition(s), training clips [1, 2, 3]")
    dms_3 = []
    for sid in sids:
        dm = get_response_data_by_rep(sid, [1, 2, 3], mask, n_reps=rep)
        assert dm.shape == (1350, 19863)
        dms_3.append(dm)
    dms_3 = np.stack(dms_3, axis=0)

    print(f"Saving template to: {out_fn}")
    nb.record(out_fn, searchlight_template)(dms_3, sls, dists, radius, n_jobs=-1)
