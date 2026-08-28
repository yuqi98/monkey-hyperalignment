import os

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment import searchlight_procrustes, searchlight_ridge
from scipy.stats import zscore

dset = nb.MonkeyKingdom()
sids = dset.subjects
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
radius = 20


def get_response_data(sid):
    dms = []
    for run in [1, 2, 3]:
        dm = dset.get_data(sid, 'monkey', run, 'lr')
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


def hyperalign_to_template(dm, tpl, lr, radius, align):
    sls = nb.sls(lr, radius, space="onavg-ico32", mask=True)
    mat0 = nb.load(f"{root}/tmp/{lr}h_{radius}mm_human.npz")
    if align == "procr":
        func = searchlight_procrustes
    elif align == "ridge":
        func = searchlight_ridge
    else:
        raise ValueError(f"Unknown alignment method: {align}")
    return func(dm, tpl, sls, sls, mat0=mat0)


if __name__ == "__main__":
    print(sids)

    mat0_fn = f"{root}/tmp/lrh_{radius}mm_human.npz"
    if not os.path.exists(mat0_fn):
        sls = nb.sls('lr', radius, space="onavg-ico32", mask=True)
        mat = initialize_sparse_matrix(sls)
        nb.save(mat0_fn, mat)

    template = np.load(f"{root}/templates/human_3_clips.npy")
    assert template.shape == (2700, 19341)

    jobs = []
    for sid in sids:
        dm = get_response_data(sid)
        assert dm.shape == (2700, 19341)
        for align in ["ridge", "procr"]:
            out_fn = f"{root}/transforms/human_{sid}_3_clips_all_human_{align}_unweighted.npy"
            if os.path.exists(out_fn):
                continue
            jobs.append(delayed(nb.record(out_fn, hyperalign_to_template))(dm, template, "lr", radius, align))

    with Parallel(n_jobs=12) as parallel:
        parallel(jobs)
