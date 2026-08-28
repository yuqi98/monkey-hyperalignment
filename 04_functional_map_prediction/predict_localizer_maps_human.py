import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuroboros as nb
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import zscore

ROOT = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data"
TRANSFORM_ROOT = f"{ROOT}/data_for_haiyan/transformations"
OUT_ROOT = f"{ROOT}/functional_maps/human_pairwise_localizer_prediction"
N_JOBS = 24

CONTRASTS = [
    [1, 0, -1, 0, 0],
    [0, 1, -1, 0, 0],
]
CONTRAST_NAMES = ["face_vs_object", "body_vs_object"]


def run_glm(i, dm, all_confounds, all_regressors, contrasts):
    confounds = all_confounds[i]
    regressors = all_regressors[i]
    beta_conf = np.linalg.lstsq(confounds, dm, rcond=None)[0]
    denoised = dm - confounds @ beta_conf
    denoised = np.nan_to_num(zscore(denoised, axis=0))
    return nb.glm(denoised, regressors, confounds, contrasts=contrasts, return_r2=True)


def predict_target(ti, align, sids, all_dms_native, transform_root, all_confounds, all_regressors, contrasts):
    n_sids = len(sids)
    tgt_sid = sids[ti]
    avg_dm = np.zeros(all_dms_native.shape[1:], dtype=np.float32)
    for si, src_sid in enumerate(sids):
        if si == ti:
            continue
        xfm = nb.load(f"{transform_root}/human_{src_sid}_to_{tgt_sid}_all_clips_{align}_with_mask.npz")
        xfm = xfm.tocsr().astype(np.float32)
        projected = all_dms_native[si] @ xfm
        avg_dm += np.nan_to_num(zscore(projected, axis=0))
    avg_dm /= (n_sids - 1)
    betas, ts, R2s = run_glm(ti, avg_dm, all_confounds, all_regressors, contrasts)
    return ti, betas, ts, R2s


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    dset = nb.MonkeyKingdom()
    sids = dset.subjects
    n_sids = len(sids)

    all_confounds = np.load(f"{ROOT}/cross_species/localizers/all_confounds.npy")
    all_regressors = np.load(f"{ROOT}/cross_species/localizers/all_regressors.npy")

    print("Loading native (full onavg-ico32 mesh) localizer data for all subjects...", flush=True)
    t0 = time.time()
    all_dms_native = []
    for sid in sids:
        dms = []
        for run in range(1, 6):
            dms.append(dset.get_data(sid, 'localizer', run, 'lr'))
        all_dms_native.append(np.concatenate(dms, axis=0).astype(np.float32))
    all_dms_native = np.array(all_dms_native)
    print(f"  loaded in {time.time() - t0:.1f}s, shape {all_dms_native.shape}", flush=True)

    print("Computing measured (actual) contrast maps...", flush=True)
    actual_ts = []
    for i in range(n_sids):
        dm = np.nan_to_num(zscore(all_dms_native[i], axis=0))
        betas, ts, R2s = run_glm(i, dm, all_confounds, all_regressors, CONTRASTS)
        actual_ts.append(ts)
    actual_ts = np.array(actual_ts)
    np.save(f"{OUT_ROOT}/actual_ts_all_subjects.npy", actual_ts)
    np.save(f"{OUT_ROOT}/sids.npy", np.array(sids))
    print("  actual_ts shape:", actual_ts.shape, flush=True)

    for c, cname in enumerate(CONTRAST_NAMES):
        nb.plot(
            np.mean(actual_ts[:, c], axis=0), cmap="bwr", vmax=15, vmin=-15,
            title=f"{cname.replace('_', ' ')} (actual, avg ts)",
            fn=f"{OUT_ROOT}/avg_actual_{cname}.png",
        )

    summary = {}
    for align in ["procr", "ridge"]:
        print(f"\n=== align = {align} ===", flush=True)
        t0 = time.time()
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(predict_target)(
                ti, align, sids, all_dms_native, TRANSFORM_ROOT, all_confounds, all_regressors, CONTRASTS
            )
            for ti in range(n_sids)
        )
        print(f"  prediction for all {n_sids} subjects took {time.time() - t0:.1f}s", flush=True)

        predicted_betas = np.zeros_like(actual_ts)
        predicted_ts = np.zeros_like(actual_ts)
        predicted_R2s = None
        for ti, betas, ts, R2s in results:
            predicted_betas[ti] = betas
            predicted_ts[ti] = ts
            if predicted_R2s is None:
                predicted_R2s = np.zeros((n_sids,) + np.array(R2s).shape)
            predicted_R2s[ti] = R2s

        np.save(f"{OUT_ROOT}/predicted_ts_{align}.npy", predicted_ts)
        np.save(f"{OUT_ROOT}/predicted_betas_{align}.npy", predicted_betas)
        np.save(f"{OUT_ROOT}/predicted_R2s_{align}.npy", predicted_R2s)

        for c, cname in enumerate(CONTRAST_NAMES):
            nb.plot(
                np.mean(predicted_ts[:, c], axis=0), cmap="bwr", vmax=15, vmin=-15,
                title=f"{cname.replace('_', ' ')} (predicted from others, avg ts, {align})",
                fn=f"{OUT_ROOT}/avg_predicted_{align}_{cname}.png",
            )

        pred_z = zscore(np.nan_to_num(predicted_ts), axis=2)
        meas_z = zscore(np.nan_to_num(actual_ts), axis=2)
        corr_mat = np.zeros((n_sids, n_sids, len(CONTRASTS)))
        for pi in range(n_sids):
            corr_mat[pi] = np.mean(pred_z[pi][None, :, :] * meas_z, axis=2)
        np.save(f"{OUT_ROOT}/corr_matrix_{align}.npy", corr_mat)

        congruent = np.stack([corr_mat[i, i] for i in range(n_sids)], axis=0)
        incongruent = np.stack(
            [np.mean([corr_mat[j, i] for j in range(n_sids) if j != i], axis=0) for i in range(n_sids)],
            axis=0,
        )
        np.save(f"{OUT_ROOT}/congruent_{align}.npy", congruent)
        np.save(f"{OUT_ROOT}/incongruent_{align}.npy", incongruent)
        summary[align] = dict(congruent=congruent, incongruent=incongruent)

        print(f"  mean congruent correlation: {congruent.mean(axis=0)}", flush=True)
        print(f"  mean incongruent correlation: {incongruent.mean(axis=0)}", flush=True)

        fig, axes = plt.subplots(len(CONTRAST_NAMES), 1, figsize=(14, 10), sharex=True)
        x = np.arange(n_sids)
        width = 0.35
        labels = [sid[-4:] for sid in sids]
        for c, (cname, ax) in enumerate(zip(CONTRAST_NAMES, axes)):
            ax.bar(x - width / 2, congruent[:, c], width, label="congruent", color="#1F78B4")
            ax.bar(x + width / 2, incongruent[:, c], width, label="incongruent", color="#E31A1C")
            ax.set_title(cname.replace("_", " "), fontsize=14)
            ax.set_ylabel("Correlation", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.legend(fontsize=10)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=90, fontsize=8)
        fig.suptitle(f"Prediction accuracy: congruent vs incongruent ({align})", fontsize=16)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/congruent_vs_incongruent_{align}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    violin_colors = ["#A6CEE3", "#FDBF6F"]
    line_colors = ["#1F78B4", "#E31A1C"]
    positions = [0, 0.4]

    for c, cname in enumerate(CONTRAST_NAMES):
        procr_vals = summary["procr"]["congruent"][:, c]
        ridge_vals = summary["ridge"]["congruent"][:, c]

        plt.figure(figsize=(5.5, 5))
        for vals, v_color, l_color, pos in zip(
            [procr_vals, ridge_vals], violin_colors, line_colors, positions
        ):
            parts = plt.violinplot(vals, positions=[pos], widths=0.3, showmeans=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(v_color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.9)
            plt.scatter(
                np.random.normal(pos, 0.02, size=len(vals)), vals,
                color="black", s=25, alpha=0.7, zorder=3,
            )
            mean_val = vals.mean()
            plt.hlines(mean_val, pos - 0.12, pos + 0.12, color=l_color, lw=3, zorder=4)
            plt.text(pos, mean_val + 0.001, f"{mean_val:.3f}", color=l_color,
                     ha="center", va="bottom", fontsize=12, fontweight="bold")

        plt.xticks(positions, ["Procrustes", "Ridge"], fontsize=15)
        plt.ylabel("Congruent correlation\n(predicted vs measured)", fontsize=16)
        plt.title(f"Individual prediction accuracy\n{cname.replace('_', ' ')}", fontsize=18, pad=15)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT_ROOT}/procr_vs_ridge_prediction_accuracy_violin_{cname}.png", dpi=300)
        plt.close()

    print("\nDone. Outputs saved under", OUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
