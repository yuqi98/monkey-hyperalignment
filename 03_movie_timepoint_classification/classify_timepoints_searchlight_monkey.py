import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkeys/nb-2s/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
radius = 20
ALIGN = "ridge"


def get_response_data(sid, clips, mask):
    dms = []
    for run in clips:
        dm = np.load(f"{DATA_ROOT}/{sid}/clip{run}_avg.npy")
        dm = dm[:, mask]
        dms.append(dm)
    return np.concatenate(dms, axis=0)


def get_hyperalignment_data(sid, clips, mask):
    dm = get_response_data(sid, clips, mask)
    transform_fn = f"{root}/transforms/monkey_{sid}_3_clips_all_monkey_{ALIGN}_with_mask_unweighted.npy"
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
    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print(sids)
    assert len(sids) == 12

    mask = nb.mask('lr', 'mkavg-ico32')
    sls = nb.sls("lr", radius, space="mkavg-ico32", mask=True)

    jobs = []
    for hyperaligned, out_tag in [
        (False, "anatomical_ridge_with_mask"),
        (True, f"hyperalign_to_template_{ALIGN}_with_mask_unweighted"),
    ]:
        get_data = get_hyperalignment_data if hyperaligned else get_response_data

        test_all = {sid: get_data(sid, [4, 5], mask) for sid in sids}
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
