import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import neuroboros as nb
import numpy as np
from scipy import sparse
from scipy.stats import zscore

MAPS_ROOT = "/dartfs/rc/lab/H/HaxbyLab/monkey_kingdom/maps"
TRANSFORM_ROOT = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/transforms"
OUT_ROOT = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/functional_maps/monkey_pairwise_localizer_prediction"
os.makedirs(OUT_ROOT, exist_ok=True)

sids = ["Dino", "Fahra", "Luce", "Marcel"]
n_sids = len(sids)

kind_names = {"allbody_obj": "body_vs_object", "allf_obj": "face_vs_object"}
kinds = list(kind_names.keys())

mask = nb.mask("lr", "mkavg-ico32")

xfms = {}
for sid in sids:
    for lr in "lr":
        xfm = nb.load(f"{MAPS_ROOT}/../feilong/xfms2/{sid}/{lr}h_native_vertices_to_mkavg-ico32_vertices.npz")
        diag = np.reciprocal(np.array(xfm.sum(axis=0)).ravel())
        xfms[sid, lr] = xfm @ sparse.diags_array(diag)

measured = {}
for sid in sids:
    for kind in kinds:
        fns = [f"{MAPS_ROOT}/localizer/{sid}/{kind}/t_{lr}h_avg0d2to0d8.mgh" for lr in "lr"]
        mm = [nib.load(fn).get_fdata().ravel() for fn in fns]
        parts = [m @ xfms[sid, lr] for m, lr in zip(mm, "lr")]
        measured[sid, kind] = np.concatenate(parts)[mask]

for kind in kinds:
    cname = kind_names[kind]
    avg_measured = np.mean([measured[sid, kind] for sid in sids], axis=0)
    result = np.full(len(mask), np.nan)
    result[mask] = avg_measured
    nb.plot_mebrains(
        result, cmap="bwr", vmax=30, vmin=-30,
        title=f"{cname.replace('_', ' ')} (actual, avg t)",
        fn=f"{OUT_ROOT}/avg_actual_{cname}.png",
    )
np.save(f"{OUT_ROOT}/actual_ts_all_subjects.npy", np.array([[measured[sid, kind] for kind in kinds] for sid in sids]))
np.save(f"{OUT_ROOT}/sids.npy", np.array(sids))

summary = {}
for align in ["procr", "ridge"]:
    print(f"=== align = {align} ===")

    predicted = {}
    for sid in sids:
        for kind in kinds:
            projected = []
            for exclude_sid in sids:
                if exclude_sid == sid:
                    continue
                transform = np.load(
                    f"{TRANSFORM_ROOT}/monkey_{exclude_sid}_to_{sid}_{align}_with_mask_unweighted.npy",
                    allow_pickle=True,
                )[()]
                new_dm = np.nan_to_num(zscore(measured[exclude_sid, kind] @ transform, axis=0))
                projected.append(new_dm)
            predicted[sid, kind] = np.mean(projected, axis=0)

    predicted_arr = np.array([[predicted[sid, kind] for kind in kinds] for sid in sids])
    np.save(f"{OUT_ROOT}/predicted_ts_{align}.npy", predicted_arr)

    for kind in kinds:
        cname = kind_names[kind]
        avg_predicted = np.mean([predicted[sid, kind] for sid in sids], axis=0)
        result = np.full(len(mask), np.nan)
        result[mask] = avg_predicted
        nb.plot_mebrains(
            result, cmap="bwr", vmax=30, vmin=-30,
            title=f"{cname.replace('_', ' ')} (predicted from others, avg t, {align})",
            fn=f"{OUT_ROOT}/avg_predicted_{align}_{cname}.png",
        )

    corr_maps = {}
    for sid in sids:
        for kind in kinds:
            p = zscore(np.nan_to_num(predicted[sid, kind]))
            for measure_sid in sids:
                m = zscore(np.nan_to_num(measured[measure_sid, kind]))
                corr_maps[sid, measure_sid, kind] = np.mean(p * m)

    congruent = np.array([[corr_maps[sid, sid, kind] for kind in kinds] for sid in sids])
    incongruent = np.array([
        [np.mean([corr_maps[other, sid, kind] for other in sids if other != sid]) for kind in kinds]
        for sid in sids
    ])
    np.save(f"{OUT_ROOT}/congruent_{align}.npy", congruent)
    np.save(f"{OUT_ROOT}/incongruent_{align}.npy", incongruent)
    summary[align] = dict(congruent=congruent, incongruent=incongruent)

    print("  congruent:\n", congruent)
    print("  incongruent:\n", incongruent)

    fig, axes = plt.subplots(1, len(kinds), figsize=(10, 5))
    x = np.arange(n_sids)
    width = 0.35
    for k, (kind, ax) in enumerate(zip(kinds, axes)):
        cname = kind_names[kind]
        ax.bar(x - width / 2, congruent[:, k], width, label="congruent", color="#1F78B4")
        ax.bar(x + width / 2, incongruent[:, k], width, label="incongruent", color="#E31A1C")
        ax.set_title(cname.replace("_", " "), fontsize=14)
        ax.set_ylabel("Correlation", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sids, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(fontsize=10)
    fig.suptitle(f"Prediction accuracy: congruent vs incongruent ({align})", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{OUT_ROOT}/congruent_vs_incongruent_{align}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

violin_colors = ["#A6CEE3", "#FDBF6F"]
line_colors = ["#1F78B4", "#E31A1C"]
positions = [0, 0.4]

for k, kind in enumerate(kinds):
    cname = kind_names[kind]
    procr_vals = summary["procr"]["congruent"][:, k]
    ridge_vals = summary["ridge"]["congruent"][:, k]

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
    plt.title(f"Individual monkey prediction accuracy\n{cname.replace('_', ' ')}", fontsize=18, pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_ROOT}/procr_vs_ridge_prediction_accuracy_violin_{cname}.png", dpi=300)
    plt.close()

print("Done. Outputs saved under", OUT_ROOT)
