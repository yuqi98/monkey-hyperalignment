import numpy as np

ISC_ROOT = "/dartfs/rc/lab/H/HaxbyLab/yuqi/monkey_kingdom_data/rep_analysis/isc"
MAX_REPS = 7
ALIGNMENTS = ["procr", "ridge"]

anatomical_avg_isc = np.full(MAX_REPS, np.nan)
for d in range(1, MAX_REPS + 1):
    isc = np.load(f"{ISC_ROOT}/anatomical_with_mask_data_rep_{d}.npy")
    anatomical_avg_isc[d - 1] = np.nanmean(isc)
np.save(f"{ISC_ROOT}/anatomical_avg_isc_by_rep.npy", anatomical_avg_isc)
print("anatomical avg ISC by data_rep (1..7):")
print(anatomical_avg_isc)

for align in ALIGNMENTS:
    grid = np.full((MAX_REPS, MAX_REPS), np.nan)
    for r in range(1, MAX_REPS + 1):
        for d in range(1, MAX_REPS + 1):
            fn = (
                f"{ISC_ROOT}/hyperaligned_3_clips_all_monkey_{align}_with_mask_unweighted"
                f"_transform_rep_{r}_data_rep_{d}.npy"
            )
            isc = np.load(fn)
            grid[r - 1, d - 1] = np.nanmean(isc)
    np.save(f"{ISC_ROOT}/{align}_avg_isc_grid.npy", grid)
    print(f"\n{align} avg ISC grid (rows=transform_rep 1..7, cols=data_rep 1..7):")
    print(grid)
