import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment import searchlight_procrustes, searchlight_ridge
from scipy.stats import zscore

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkeys/nb-2s/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
radius = 20


def get_response_data(sid, mask):
    dms = []
    for run in [1, 2, 3, 4, 5]:
        dm = np.load(f"{DATA_ROOT}/{sid}/clip{run}_avg.npy")
        dm = dm[:, mask]
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


def hyperalign_to_subject(dm, tpl, lr, radius, align):
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
    sids = ['Dino', 'Fahra', 'Luce', 'Marcel']

    mat0_fn = f"{root}/tmp/lrh_{radius}mm_with_mask.npz"
    if not os.path.exists(mat0_fn):
        sls = nb.sls('lr', radius, space="mkavg-ico32", mask=True)
        mat = initialize_sparse_matrix(sls)
        nb.save(mat0_fn, mat)

    mask = nb.mask('lr', 'mkavg-ico32')
    data = {sid: get_response_data(sid, mask) for sid in sids}

    jobs = []
    for sid in sids:
        for exclude_sid in sids:
            if sid == exclude_sid:
                continue
            for align in ["procr", "ridge"]:
                out_fn = f"{root}/transforms/monkey_{sid}_to_{exclude_sid}_{align}_with_mask_unweighted.npy"
                if os.path.exists(out_fn):
                    continue
                jobs.append(delayed(nb.record(out_fn, hyperalign_to_subject))(data[exclude_sid], data[sid], "lr", radius, align))

    print(f"{len(jobs)} pairwise transform jobs to run")
    with Parallel(n_jobs=10) as parallel:
        parallel(jobs)
