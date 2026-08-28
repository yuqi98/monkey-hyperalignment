import os
from glob import glob

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkeys/nb-2s/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
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


def get_pred_baseline(sids, sid_avg, test_clips, mask, hyperaligned, exclude_extra=None):
    dms = []
    for sid in sids:
        if sid == sid_avg or sid == exclude_extra:
            continue
        if hyperaligned:
            dm = get_hyperalignment_data(sid, test_clips, mask)
        else:
            dm = get_response_data(sid, test_clips, mask)
        dms.append(dm)
    return np.mean(dms, axis=0)


def classify(sid, train, test, pred):
    eps = 1e-8
    U, s, Vt = nb.linalg.safe_svd(train)
    reconstruction_error = np.linalg.norm(train - (U @ np.diag(s) @ Vt))
    assert reconstruction_error < eps, f"Reconstruction error is too high: {reconstruction_error}"
    U_test = (test @ Vt.T) * np.reciprocal(s + eps)[np.newaxis, :]
    U_pred = (pred @ Vt.T) * np.reciprocal(s + eps)[np.newaxis, :]
    npcs = np.arange(10, min(U_test.shape[0], 900) + 1, 10)
    accs = [nb.benchmark.classification(U_test, U_pred, npc=npc) for npc in npcs]
    return np.array(accs)


if __name__ == "__main__":
    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print(sids)
    assert len(sids) == 12

    mask = nb.mask('lr', 'mkavg-ico32')

    train_anat = {sid: get_response_data(sid, [1, 2, 3], mask) for sid in sids}
    test_anat = {sid: get_response_data(sid, [4, 5], mask) for sid in sids}
    train_hyper = {sid: get_hyperalignment_data(sid, [1, 2, 3], mask) for sid in sids}
    test_hyper = {sid: get_hyperalignment_data(sid, [4, 5], mask) for sid in sids}

    jobs = []

    for sid in sids:
        for exclude_sid in sids:
            if sid == exclude_sid:
                continue

            out_fn = f"{root}/classification/anatomical_train/{sid}_with_mask_exclude_{exclude_sid}.npy"
            if not os.path.exists(out_fn):
                pred = get_pred_baseline(sids, sid, [4, 5], mask, hyperaligned=False, exclude_extra=exclude_sid)
                jobs.append(delayed(nb.record(out_fn, classify))(sid, train_anat[sid], test_anat[sid], pred))

            out_fn = f"{root}/classification/hyperaligned_train/{sid}_all_{ALIGN}_with_mask_unweighted_exclude_{exclude_sid}.npy"
            if not os.path.exists(out_fn):
                pred = get_pred_baseline(sids, sid, [4, 5], mask, hyperaligned=True, exclude_extra=exclude_sid)
                jobs.append(delayed(nb.record(out_fn, classify))(sid, train_hyper[sid], test_hyper[sid], pred))

    for sid in sids:
        out_fn = f"{root}/classification/anatomical/{sid}_with_mask_latest.npy"
        if not os.path.exists(out_fn):
            pred = get_pred_baseline(sids, sid, [4, 5], mask, hyperaligned=False)
            jobs.append(delayed(nb.record(out_fn, classify))(sid, train_anat[sid], test_anat[sid], pred))

        out_fn = f"{root}/classification/hyperaligned/{sid}_all_{ALIGN}_with_mask_unweighted_latest.npy"
        if not os.path.exists(out_fn):
            pred = get_pred_baseline(sids, sid, [4, 5], mask, hyperaligned=True)
            jobs.append(delayed(nb.record(out_fn, classify))(sid, train_hyper[sid], test_hyper[sid], pred))

    print(f"{len(jobs)} classification jobs to run")
    with Parallel(n_jobs=24) as parallel:
        parallel(jobs)
