import os

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed

root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
radius = 20
ALIGN = "ridge"


def get_response_data(dset, sid, clips):
    dms = []
    for run in clips:
        dm = dset.get_data(sid, 'monkey', run, 'lr')
        dms.append(dm)
    return np.concatenate(dms, axis=0)


def get_hyperalignment_data(dset, sid, clips):
    dm = get_response_data(dset, sid, clips)
    transform_fn = f"{root}/transforms/human_{sid}_3_clips_all_human_{ALIGN}_unweighted.npy"
    transform = np.load(transform_fn, allow_pickle=True)
    return dm @ transform[()]


def classify_by_searchlight(sid, pred_all, test_all, sls):
    accs = []
    for sl in sls:
        dm_pred = pred_all[sid][:, sl]
        dm_test = test_all[sid][:, sl]
        accs.append(nb.benchmark.classification(dm_test, dm_pred))
    return np.array(accs)


if __name__ == "__main__":
    dset = nb.MonkeyKingdom()
    sids = dset.subjects
    print(sids)
    assert len(sids) == 24

    sls = nb.sls("lr", radius, space="onavg-ico32", mask=True)

    jobs = []
    for hyperaligned, out_tag in [
        (False, "anatomical_ridge_human"),
        (True, f"hyperalign_to_template_{ALIGN}_human_unweighted"),
    ]:
        get_data = get_hyperalignment_data if hyperaligned else get_response_data

        test_all = {sid: get_data(dset, sid, [4, 5]) for sid in sids}
        pred_all = {
            sid: np.mean([test_all[other] for other in sids if other != sid], axis=0)
            for sid in sids
        }

        for sid in sids:
            out_fn = f"{root}/classification/searchlight_analysis/{sid}_{out_tag}_timepoint_latest.npy"
            if os.path.exists(out_fn):
                continue
            jobs.append(delayed(nb.record(out_fn, classify_by_searchlight))(sid, pred_all, test_all, sls))

    print(f"{len(jobs)} searchlight classification jobs to run")
    with Parallel(n_jobs=12) as parallel:
        parallel(jobs)
