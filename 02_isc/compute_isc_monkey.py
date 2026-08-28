import os
from glob import glob

import neuroboros as nb
import numpy as np
from scipy.stats import zscore

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkeys/nb-2s/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"


def get_response_data(sid, mask):
    dms = []
    for run in [4, 5]:
        dm = np.load(f"{DATA_ROOT}/{sid}/clip{run}_avg.npy")
        dm = dm[:, mask]
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


def compute_isc(all_data, sids, align=None):
    dms = []
    for sid in sids:
        dm = all_data[sid]
        if align is not None:
            transform = np.load(
                f"{root}/transforms/monkey_{sid}_3_clips_all_monkey_{align}_with_mask_unweighted.npy",
                allow_pickle=True,
            )
            dm = np.nan_to_num(zscore(dm @ transform[()], axis=0))
        dms.append(dm)
    dms = np.stack(dms, axis=0)
    return nb.isc(dms, pairwise=False, metric='correlation')


if __name__ == "__main__":
    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print(sids)
    assert len(sids) == 12

    mask = nb.mask('lr', 'mkavg-ico32')
    all_data = {}
    for sid in sids:
        dm = get_response_data(sid, mask)
        assert dm.shape == (900, 19863)
        all_data[sid] = dm

    anatomical_fn = f"{root}/isc/anatomical_with_mask.npy"
    nb.record(anatomical_fn, compute_isc)(all_data, sids)

    for align in ["procr", "ridge"]:
        hyperaligned_fn = f"{root}/isc/hyperaligned_3_clips_all_monkey_{align}_with_mask_unweighted.npy"
        nb.record(hyperaligned_fn, compute_isc)(all_data, sids, align=align)
