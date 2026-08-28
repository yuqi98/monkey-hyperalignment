import neuroboros as nb
import numpy as np
from scipy.stats import zscore
from hyperalignment import searchlight_template

dset = nb.MonkeyKingdom()
sids = dset.subjects
root = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/templates"
radius = 20


def get_response_data(sid, runs):
    dms = []
    for run in runs:
        dm = dset.get_data(sid, 'monkey', run, 'lr')
        dms.append(dm)
    dms = np.nan_to_num(zscore(np.concatenate(dms, axis=0), axis=0))
    return dms


if __name__ == "__main__":
    print(sids)

    dms_3 = []
    for sid in sids:
        dm = get_response_data(sid, [1, 2, 3])
        assert dm.shape == (2700, 19341)
        dms_3.append(dm)
    dms_3 = np.stack(dms_3, axis=0)

    sls, dists = nb.sls("lr", radius, space="onavg-ico32", return_dists=True, mask=True)
    out_fn_3 = f"{root}/human_3_clips.npy"
    nb.record(out_fn_3, searchlight_template)(dms_3, sls, dists, radius, n_jobs=-1)
