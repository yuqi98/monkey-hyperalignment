import os
from glob import glob

import neuroboros as nb
import numpy as np
from scipy.stats import zscore
from hyperalignment import searchlight_template

DATA_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/feilong/data/monkeys/nb-2s/mkavg-ico32"
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/templates"
radius = 20


def get_response_data(sid, runs, mask):
    dms = []
    for run in runs:
        dm = np.load(f"{DATA_ROOT}/{sid}/clip{run}_avg.npy")
        dm = dm[:, mask]
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


if __name__ == "__main__":
    folders = sorted(glob(f"{DATA_ROOT}/*"))
    sids = [os.path.basename(_) for _ in folders]
    print(sids)
    assert len(sids) == 12

    mask = nb.mask('lr', 'mkavg-ico32')
    sls, dists = nb.sls("lr", radius, space="mkavg-ico32", return_dists=True, mask=True)
    assert len(sls) == 19863

    dms_3 = []
    for sid in sids:
        dm = get_response_data(sid, [1, 2, 3], mask)
        assert dm.shape == (1350, 19863)
        dms_3.append(dm)
    dms_3 = np.stack(dms_3, axis=0)

    out_fn_3 = f"{root}/monkey_3_clips_with_mask.npy"
    nb.record(out_fn_3, searchlight_template)(dms_3, sls, dists, radius, n_jobs=-1)
