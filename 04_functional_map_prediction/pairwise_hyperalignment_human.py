import os
import sys
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import zscore
from hyperalignment.sparse import initialize_sparse_matrix
from hyperalignment import searchlight_weights, searchlight_procrustes, searchlight_ridge

root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
transform_out_root = f"{root}/data_for_haiyan/transformations"

dset = nb.MonkeyKingdom()
sids_human = dset.subjects
radius = 20

def get_response_data_human(sid, runs):
    dms = []
    for run in runs:
        dm = dset.get_data(sid, 'monkey', run, 'lr')
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms

def hyperalign_pairwise(src_sid, tgt_sid, active_runs, lr, radius, align, space, mat0_fn, expected_timepoints, expected_vertices):
    dm_src = get_response_data_human(src_sid, active_runs)
    dm_tgt = get_response_data_human(tgt_sid, active_runs)

    assert dm_src.shape == (expected_timepoints, expected_vertices), f"Source shape mismatch for {src_sid}!"
    assert dm_tgt.shape == (expected_timepoints, expected_vertices), f"Target shape mismatch for {tgt_sid}!"
    
    sls, dists = nb.sls(lr, radius, space=space, mask=True, return_dists=True)
    weights = searchlight_weights(sls, None, radius)
    mat0 = nb.load(mat0_fn)
    
    if align == "procr":
        func = searchlight_procrustes
    elif align == "ridge":
        func = searchlight_ridge
    else:
        raise ValueError(f"Unknown alignment method: {align}")
        
    xfm = func(dm_src, dm_tgt, sls, sls, mat0=mat0, weights=weights)
    return xfm


if __name__ == "__main__":
    os.makedirs(transform_out_root, exist_ok=True)
    os.makedirs(f"{root}/tmp", exist_ok=True)

    mat0_human_fn = f"{root}/tmp/lrh_{radius}mm_human.npz"
    if not os.path.exists(mat0_human_fn):
        sls_h = nb.sls('lr', radius, space="onavg-ico32", mask=True)
        mat_h = initialize_sparse_matrix(sls_h)
        nb.save(mat0_human_fn, mat_h)

    active_runs = [1, 2, 3, 4, 5]
    config_str = "all_clips"
    alignments = ["ridge", "procr"]

    space = "onavg-ico32"
    mat0_fn = mat0_human_fn
    expected_vertices = 19341
    expected_timepoints = len(active_runs) * 900

    jobs = []

    print(f"Queuing human pairwise alignment jobs for configuration: {config_str}")

    for src_sid in sids_human:
        for tgt_sid in sids_human:
            if src_sid == tgt_sid:
                continue

            for align in alignments:
                out_fn = f"{transform_out_root}/human_{src_sid}_to_{tgt_sid}_{config_str}_{align}_with_mask.npz"

                if os.path.exists(out_fn):
                    continue
                    
                jobs.append(
                    delayed(nb.record(out_fn, hyperalign_pairwise))(
                        src_sid, tgt_sid, active_runs, "lr", radius, align, space, mat0_fn, expected_timepoints, expected_vertices
                    )
                )

    print(f"\nTotal pairwise transformation jobs ready for calculation: {len(jobs)}")
    with Parallel(n_jobs=30) as parallel:
        parallel(jobs)