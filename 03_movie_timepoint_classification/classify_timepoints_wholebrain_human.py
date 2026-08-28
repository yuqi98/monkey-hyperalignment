import os

import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed

root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
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


def get_pred_baseline(sids, test_cache, sid_avg, exclude_extra=None):
    dms = [test_cache[sid] for sid in sids if sid != sid_avg and sid != exclude_extra]
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
    dset = nb.MonkeyKingdom()
    sids = dset.subjects
    print(sids)
    assert len(sids) == 24

    train_anat = {sid: get_response_data(dset, sid, [1, 2, 3]) for sid in sids}
    test_anat = {sid: get_response_data(dset, sid, [4, 5]) for sid in sids}
    train_hyper = {sid: get_hyperalignment_data(dset, sid, [1, 2, 3]) for sid in sids}
    test_hyper = {sid: get_hyperalignment_data(dset, sid, [4, 5]) for sid in sids}

    jobs = []

    for sid in sids:
        for exclude_sid in sids:
            if sid == exclude_sid:
                continue

            out_fn = f"{root}/classification/anatomical_train/{sid}_all_human_exclude_{exclude_sid}.npy"
            if not os.path.exists(out_fn):
                pred = get_pred_baseline(sids, test_anat, sid, exclude_extra=exclude_sid)
                jobs.append(delayed(nb.record(out_fn, classify))(sid, train_anat[sid], test_anat[sid], pred))

            out_fn = f"{root}/classification/hyperaligned_train/{sid}_all_human_{ALIGN}_unweighted_exclude_{exclude_sid}.npy"
            if not os.path.exists(out_fn):
                pred = get_pred_baseline(sids, test_hyper, sid, exclude_extra=exclude_sid)
                jobs.append(delayed(nb.record(out_fn, classify))(sid, train_hyper[sid], test_hyper[sid], pred))

    for sid in sids:
        out_fn = f"{root}/classification/anatomical/{sid}_human_latest.npy"
        if not os.path.exists(out_fn):
            pred = get_pred_baseline(sids, test_anat, sid)
            jobs.append(delayed(nb.record(out_fn, classify))(sid, train_anat[sid], test_anat[sid], pred))

        out_fn = f"{root}/classification/hyperaligned/{sid}_all_human_{ALIGN}_unweighted_latest.npy"
        if not os.path.exists(out_fn):
            pred = get_pred_baseline(sids, test_hyper, sid)
            jobs.append(delayed(nb.record(out_fn, classify))(sid, train_hyper[sid], test_hyper[sid], pred))

    print(f"{len(jobs)} classification jobs to run")
    with Parallel(n_jobs=24) as parallel:
        parallel(jobs)
