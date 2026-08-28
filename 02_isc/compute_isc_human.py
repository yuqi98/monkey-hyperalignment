import neuroboros as nb
import numpy as np
from scipy.stats import zscore

root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"


def get_response_data(dset, sid):
    dms = []
    for run in [4, 5]:
        dm = dset.get_data(sid, 'monkey', run, 'lr')
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


def compute_isc(all_data, sids, align=None):
    dms = []
    for sid in sids:
        dm = all_data[sid]
        if align is not None:
            transform = np.load(
                f"{root}/transforms/human_{sid}_3_clips_all_human_{align}_unweighted.npy",
                allow_pickle=True,
            )
            dm = np.nan_to_num(zscore(dm @ transform[()], axis=0))
        dms.append(dm)
    dms = np.stack(dms, axis=0)
    return nb.isc(dms, pairwise=False, metric='correlation')


if __name__ == "__main__":
    dset = nb.MonkeyKingdom()
    sids = dset.subjects
    print(sids)
    assert len(sids) == 24

    all_data = {}
    for sid in sids:
        dm = get_response_data(dset, sid)
        assert dm.shape == (1800, 19341)
        all_data[sid] = dm

    anatomical_fn = f"{root}/isc/anatomical_human.npy"
    nb.record(anatomical_fn, compute_isc)(all_data, sids)

    for align in ["procr", "ridge"]:
        hyperaligned_fn = f"{root}/isc/hyperaligned_3_clips_all_human_{align}_unweighted.npy"
        nb.record(hyperaligned_fn, compute_isc)(all_data, sids, align=align)
