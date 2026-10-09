"""Temperature scaling: does the textbook post-hoc calibration fix recover confidence under
degradation? Fit one T on clean (what you'd have in-distribution at calibration time), apply it
everywhere, and compare D-ECE. Oracle per-condition T is the unachievable upper bound."""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.metrics import d_ece

df = pd.read_csv(os.path.join(HERE, "results_thermal", "records.csv"))
PERTS = ["atmospheric_blur", "range_detail_loss", "sensor_noise", "motion_blur", "low_contrast"]
EPS = 1e-6

def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def applyT(s, T): return 1.0 / (1.0 + np.exp(-logit(s) / T))
def bce(s, y): p = np.clip(s, EPS, 1 - EPS); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
def fitT(s, y): return minimize_scalar(lambda T: bce(applyT(s, T), y), bounds=(0.05, 20.0), method="bounded").x

clean = df[(df.perturbation == "clean") & (df.severity == 0.0)]
cs, cy = clean.score.to_numpy(), clean.is_tp.to_numpy()
T_clean = fitT(cs, cy)
print(f"T fit on clean (minimising NLL) = {T_clean:.3f}")
print(f"clean D-ECE: raw {d_ece(cs, cy)[0]:.3f} -> after clean-T {d_ece(applyT(cs, T_clean), cy)[0]:.3f}\n")

print(f"{'condition @max':18s} {'raw':>6s} {'clean-T':>8s} {'oracle-T':>9s} {'T*':>6s}")
rows = []
for p in PERTS:
    c = df[(df.perturbation == p) & (df.severity == 1.0)]
    s, y = c.score.to_numpy(), c.is_tp.to_numpy()
    raw = d_ece(s, y)[0]
    ct = d_ece(applyT(s, T_clean), y)[0]
    To = fitT(s, y); orc = d_ece(applyT(s, To), y)[0]
    rows.append((p, raw, ct, orc, To))
    print(f"{p:18s} {raw:6.3f} {ct:8.3f} {orc:9.3f} {To:6.2f}")

# averages over the five degradations
import numpy as np
arr = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"\nmean over 5 degradations:  raw {arr[:,0].mean():.3f}  clean-T {arr[:,1].mean():.3f}  oracle-T {arr[:,2].mean():.3f}")
print(f"clean-T reduces degraded D-ECE by {100*(1-arr[:,1].mean()/arr[:,0].mean()):.0f}%; "
      f"oracle-T by {100*(1-arr[:,2].mean()/arr[:,0].mean()):.0f}%")
