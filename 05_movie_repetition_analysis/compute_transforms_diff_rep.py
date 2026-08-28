import os
import sys
from glob import glob

import neuroboros as nb
import numpy as np
from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment import searchlight_procrustes, searchlight_ridge
from scipy.stats import zscore

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkey/nb/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
OUT_ROOT = f"{root}/rep_analysis"
TEMPLATE_ROOT = f"{OUT_ROOT}/templates"
TRANSFORM_ROOT = f"{OUT_ROOT}/transforms"
radius = 20
MAX_REPS = 7
ALIGNMENTS = ["ridge", "procr"]


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


def hyperalign_to_template(dm, tpl, lr, radius, align):
    sls = nb.sls(lr, radius, space="mkavg-ico32", mask=True)
    mat0 = nb.load(f"{root}/tmp/{lr}h_{radius}mm_with_mask.npz")
    if align == "procr":
        func = searchlight_procrustes
    elif align == "ridge":
        func = searchlight_ridge
    else:
        raise ValueError(f"Unknown alignment method: {align}")
    return func(dm, tpl, sls, sls, mat0=mat0)


if __name__ == "__main__":
    os.makedirs(TRANSFORM_ROOT, exist_ok=True)

    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print("Subjects found:", sids)
    assert len(sids) == 12

    mat0_fn = f"{root}/tmp/lrh_{radius}mm_with_mask.npz"
    if not os.path.exists(mat0_fn):
        sls = nb.sls("lr", radius, space="mkavg-ico32", mask=False)
        mat = initialize_sparse_matrix(sls)
        nb.save(mat0_fn, mat)

    mask = nb.mask("lr", "mkavg-ico32")

    combos = [
        (rep, sid, align)
        for rep in range(1, MAX_REPS + 1)
        for sid in sids
        for align in ALIGNMENTS
    ]
    arg = int(sys.argv[1])
    rep, sid, align = combos[arg]
    print(f"rep={rep} sid={sid} align={align}")

    out_fn = f"{TRANSFORM_ROOT}/monkey_{sid}_3_clips_all_monkey_{align}_with_mask_unweighted_rep_{rep}.npy"
    if os.path.exists(out_fn):
        print(f"{out_fn} already exists, skipping")
        sys.exit(0)

    template = np.load(f"{TEMPLATE_ROOT}/monkey_3_clips_with_mask_rep_{rep}.npy")
    assert template.shape == (1350, 19863)

    dm = get_response_data_by_rep(sid, [1, 2, 3], mask, n_reps=rep)
    assert dm.shape == (1350, 19863)

    print(f"Saving transform to: {out_fn}")
    nb.record(out_fn, hyperalign_to_template)(dm, template, "lr", radius, align)
