

from __future__ import annotations

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

from simulator_wrapper import (
    Simulator, ScenarioConfig, CostConfig,
    Vmin_lim, Vmax_lim, SOC_MIN, SOC_MAX,
)


PATH_LINES_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\newlineparameters.xlsx"

PATH_LOAD_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\CIGRE_15day_Loads_R1toR18_CLEAN_S95cap.xlsx"
SHEET_PLOAD     = "Active_kW_R1toR18"
SHEET_QLOAD     = "Reactive_kVAr_R1toR18"

PATH_PV_XLSX    = r"C:\Users\Dell\Desktop\GRT\Codes\PV_15day_Hourly_Profile_Khumaltar_NOCT.xlsx"
SHEET_PV        = "PV_Hourly_15days"
PV_PER_KW_COL   = "PV_kW_per_1kW_STC"


RESULTS_DIR = r"C:\Users\Dell\Desktop\GRT\Codes\Stage2_NSAGII\Results\n20_convergence"


POP_SIZE = 60
N_GEN = 60
SEED_OPT = 7
SEED_VAL = 77


N_SCEN_OPT = 20
N_SCEN_VAL = 20


W_V = 5e6
W_EXPORT = 2e6
W_SOC = 1e6
W_VIOLBH = 2e5


EXPORT_TOL_KWH = 1e-3
FC_PMIN_FRAC = 0.0


EPS_RELAX = 0.05


XL = np.zeros(7)
XU = 2.0 * np.ones(7)


ALT_BASE_ARCHS = {
    "A1":  {"PV16_kW": 10.0, "PV17_kW": 3.0, "WT16_kW": 0.0,  "BESS_P_kW": 0.0,  "BESS_E_kWh": 0.0,   "MT_kW": 30.0, "FC_kW": 0.0},
    "A2":  {"PV16_kW": 10.0, "PV17_kW": 3.0, "WT16_kW": 10.0, "BESS_P_kW": 0.0,  "BESS_E_kWh": 0.0,   "MT_kW": 30.0, "FC_kW": 0.0},
    "A14": {"PV16_kW": 20.0, "PV17_kW": 6.0, "WT16_kW": 15.0, "BESS_P_kW": 30.0, "BESS_E_kWh": 120.0, "MT_kW": 30.0, "FC_kW": 10.0},
}


def _set_ieee_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    })


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def knee_point(F: np.ndarray) -> int:
    F = np.asarray(F, dtype=float)
    fmin = F.min(axis=0)
    fmax = F.max(axis=0)
    denom = np.where((fmax - fmin) < 1e-12, 1.0, (fmax - fmin))
    Fn = (F - fmin) / denom
    d = np.sqrt((Fn ** 2).sum(axis=1))
    return int(np.argmin(d))


def epsilon_pick(F: np.ndarray, relax: float = 0.05) -> int:
    F = np.asarray(F, dtype=float)
    imp = F[:, 0]
    cost = F[:, 1]
    imp_min = float(np.min(imp))
    thr = imp_min * (1.0 + float(relax))
    cand = np.where(imp <= thr)[0]
    if cand.size == 0:
        return int(np.argmin(cost))
    return int(cand[np.argmin(cost[cand])])


def robust_ok_from_metrics(met: dict) -> dict:
    ok_v = (met["Vmin_pu_worst"] >= Vmin_lim - 1e-9) and (met["Vmax_pu_worst"] <= Vmax_lim + 1e-9)
    ok_export = (met["Export_kWh_worst"] <= EXPORT_TOL_KWH + 1e-12)
    ok_soc = (met["SOC_min_worst"] >= SOC_MIN - 1e-9) and (met["SOC_max_worst"] <= SOC_MAX + 1e-9)
    ok_viol = (met["ViolBH_worst"] <= 0.0 + 1e-9)
    return {
        "val_robust_ok_v": int(ok_v),
        "val_robust_ok_export": int(ok_export),
        "val_robust_ok_soc": int(ok_soc),
        "val_robust_ok_violbh": int(ok_viol),
        "val_robust_ok_all": int(ok_v and ok_export and ok_soc and ok_viol),
    }


def soft_penalty(met: dict) -> float:
    v_pen = max(0.0, Vmin_lim - met["Vmin_pu_worst"]) + max(0.0, met["Vmax_pu_worst"] - Vmax_lim)
    export_pen = max(0.0, met["Export_kWh_worst"] - EXPORT_TOL_KWH)
    soc_pen = max(0.0, SOC_MIN - met["SOC_min_worst"]) + max(0.0, met["SOC_max_worst"] - SOC_MAX)
    violbh_pen = max(0.0, met["ViolBH_worst"])

    return (W_V * v_pen) + (W_EXPORT * export_pen) + (W_SOC * soc_pen) + (W_VIOLBH * violbh_pen)


def make_problem(sim: Simulator, alt_id: str, base_arch: dict, scen_opt: ScenarioConfig, cost_cfg: CostConfig):
    from pymoo.core.problem import Problem

    class Stage2Problem(Problem):
        def __init__(self):
            super().__init__(n_var=7, n_obj=2, n_constr=0, xl=XL, xu=XU)

        def _evaluate(self, X, out, *args, **kwargs):
            F = np.zeros((X.shape[0], 2), dtype=float)

            for i in range(X.shape[0]):
                obj, met, _ = sim.evaluate_architecture(
                    x=X[i, :],
                    base_arch=base_arch,
                    scenario_cfg=scen_opt,
                    cost_cfg=cost_cfg,
                    arch_id=f"{alt_id}_cand_{i}",
                    n_workers=0,
                    export_tol_kwh=EXPORT_TOL_KWH,
                    fc_pmin_frac=FC_PMIN_FRAC
                )

                pen = soft_penalty(met)
                F[i, 0] = float(obj["Import_kWh_mean"]) + pen
                F[i, 1] = float(obj["TotalCost_$yr_mean"]) + pen

            out["F"] = F

    return Stage2Problem()


def save_pareto_excel(path: str, X: np.ndarray, F_raw: np.ndarray, mets: list, alt_id: str,
                      knee_idx: int, eps_idx: int):
    dfX = pd.DataFrame(X, columns=["s_PV16", "s_PV17", "s_WT16", "s_BESS_P", "s_BESS_E", "s_MT", "s_FC"])
    dfF = pd.DataFrame(F_raw, columns=["Import_kWh_mean", "TotalCost_$yr_mean"])
    dfM = pd.DataFrame(mets)
    df = pd.concat([dfX, dfF, dfM], axis=1)
    df["pick_knee"] = 0
    df["pick_eps"] = 0
    df.loc[knee_idx, "pick_knee"] = 1
    df.loc[eps_idx, "pick_eps"] = 1
    df.insert(0, "AltID", alt_id)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Pareto", index=False)


def save_validation_excel(path: str, alt_id: str, sol_name: str, df_per: pd.DataFrame, met: dict):
    df = df_per.copy()
    df.insert(0, "AltID", alt_id)
    df.insert(1, "Solution", sol_name)

    summary = pd.DataFrame([met])
    summary.insert(0, "AltID", alt_id)
    summary.insert(1, "Solution", sol_name)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="PerScenario", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)


def plot_pareto(path_png: str, path_pdf: str, F: np.ndarray, knee_point_xy: tuple, eps_point_xy: tuple, title: str):
    _set_ieee_plot_style()
    fig = plt.figure(figsize=(3.5, 2.6))
    ax = fig.add_subplot(111)
    ax.scatter(F[:, 0], F[:, 1], s=18)
    ax.scatter([knee_point_xy[0]], [knee_point_xy[1]], marker="^", s=55)
    ax.scatter([eps_point_xy[0]], [eps_point_xy[1]], marker="x", s=55)

    ax.set_xlabel("Grid import (kWh) — mean")
    ax.set_ylabel("Annualized cost ($/yr) — mean")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path_png, dpi=600, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def _build_runinfo_row(alt_id: str, runtime_s: float, pareto_size: int) -> dict:
    return {
        "AltID": alt_id,
        "N_SCEN_OPT": N_SCEN_OPT,
        "N_SCEN_VAL": N_SCEN_VAL,
        "SEED_OPT": SEED_OPT,
        "SEED_VAL": SEED_VAL,
        "POP_SIZE": POP_SIZE,
        "N_GEN": N_GEN,
        "EXPORT_TOL_KWH": EXPORT_TOL_KWH,
        "FC_PMIN_FRAC": FC_PMIN_FRAC,
        "W_V": W_V,
        "W_EXPORT": W_EXPORT,
        "W_SOC": W_SOC,
        "W_VIOLBH": W_VIOLBH,
        "Runtime_s": runtime_s,
        "Pareto_size": pareto_size,
    }


def run_alt(sim: Simulator, alt_id: str, base_arch: dict, stamp_batch: str) -> Tuple[List[dict], dict]:
    ensure_dir(RESULTS_DIR)

    print(f"\n{'='*90}")
    print(f"Running Stage-2 (soft) for {alt_id}  |  Opt scenarios: {N_SCEN_OPT}  |  Valid scenarios: {N_SCEN_VAL}")
    print("Base arch:", base_arch)

    scen_opt = ScenarioConfig(N=N_SCEN_OPT, seed=SEED_OPT)
    scen_val = ScenarioConfig(N=N_SCEN_VAL, seed=SEED_VAL)
    cost_cfg = CostConfig()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    problem = make_problem(sim, alt_id, base_arch, scen_opt, cost_cfg)

    algo = NSGA2(pop_size=POP_SIZE)
    termination = get_termination("n_gen", N_GEN)

    t0 = time.time()
    res = minimize(problem, algo, termination=termination, seed=SEED_OPT, verbose=True)
    t1 = time.time()

    X = np.asarray(res.X, dtype=float)
    mets = []
    F_raw = np.zeros((X.shape[0], 2), dtype=float)

    for i in range(X.shape[0]):
        obj, met, _ = sim.evaluate_architecture(
            x=X[i, :],
            base_arch=base_arch,
            scenario_cfg=scen_opt,
            cost_cfg=cost_cfg,
            arch_id=f"{alt_id}_pareto_{i}",
            n_workers=0,
            export_tol_kwh=EXPORT_TOL_KWH,
            fc_pmin_frac=FC_PMIN_FRAC
        )
        F_raw[i, 0] = obj["Import_kWh_mean"]
        F_raw[i, 1] = obj["TotalCost_$yr_mean"]
        mets.append(met)

    k_idx = knee_point(F_raw)
    e_idx = epsilon_pick(F_raw, relax=EPS_RELAX)

    print("\n--- Post-selection (soft) ---")
    print(f"Knee point: Import={F_raw[k_idx,0]:.3f}  Cost={F_raw[k_idx,1]:.3f}")
    print(f"Epsilon pick: Import={F_raw[e_idx,0]:.3f}  Cost={F_raw[e_idx,1]:.3f}  (relax={EPS_RELAX})")
    print(f"Runtime: {t1 - t0:.1f} s | Pareto size: {X.shape[0]}")

    
    sel = [("KNEE", k_idx), ("EPS", e_idx)]
    seen = set()
    val_rows = []

    for name, idx in sel:
        if idx in seen:
            continue
        seen.add(idx)

        x_sel = X[idx, :]
        _, met_v, df_per = sim.evaluate_architecture(
            x=x_sel,
            base_arch=base_arch,
            scenario_cfg=scen_val,
            cost_cfg=cost_cfg,
            arch_id=f"{alt_id}_{name}_VAL",
            n_workers=0,
            export_tol_kwh=EXPORT_TOL_KWH,
            fc_pmin_frac=FC_PMIN_FRAC
        )
        ok = robust_ok_from_metrics(met_v)
        met_v.update(ok)

        val_path = os.path.join(RESULTS_DIR, f"NSGA2_{alt_id}_ROBUST_VALIDATION_N20_{stamp_batch}_{name}.xlsx")
        save_validation_excel(val_path, alt_id, name, df_per, met_v)

        print(f"\n--- Robust validation: {alt_id} {name} ---")
        print(f"Vmin_worst={met_v['Vmin_pu_worst']:.4f}  Vmax_worst={met_v['Vmax_pu_worst']:.4f}  "
              f"Export_worst={met_v['Export_kWh_worst']:.6f}  ViolBH_worst={met_v['ViolBH_worst']:.0f}")
        print("Robust flags:", {k: met_v[k] for k in ok.keys()})
        print("Saved validation:", val_path)

        row = {
            "AltID": alt_id,
            "ScenarioCount": N_SCEN_OPT,
            "Selection": name,
            "ParetoIndex": int(idx),
            "Import_kWh_mean": float(F_raw[idx, 0]),
            "TotalCost_$yr_mean": float(F_raw[idx, 1]),
            "Runtime_s": float(t1 - t0),
            "Pareto_size": int(X.shape[0]),
            "ValidationFile": val_path,
        }

      
        for key in [
            "Vmin_pu_worst", "Vmax_pu_worst", "Export_kWh_worst", "ViolBH_worst",
            "SOC_min_worst", "SOC_max_worst",
            "Import_kWh_worst", "TotalCost_$yr_worst",
            "val_robust_ok_v", "val_robust_ok_export", "val_robust_ok_soc",
            "val_robust_ok_violbh", "val_robust_ok_all"
        ]:
            row[key] = met_v.get(key, np.nan)

        val_rows.append(row)

  
    pareto_path = os.path.join(RESULTS_DIR, f"NSGA2_SOFT_{alt_id}_Pareto_N20_{stamp_batch}.xlsx")
    save_pareto_excel(pareto_path, X, F_raw, mets, alt_id, k_idx, e_idx)
    print(f"\n✅ Finished {alt_id} | Pareto saved: {pareto_path}")

    png = os.path.join(RESULTS_DIR, f"Pareto_{alt_id}_N20_{stamp_batch}.png")
    pdf = os.path.join(RESULTS_DIR, f"Pareto_{alt_id}_N20_{stamp_batch}.pdf")
    plot_pareto(
        png, pdf, F_raw,
        knee_point_xy=(F_raw[k_idx, 0], F_raw[k_idx, 1]),
        eps_point_xy=(F_raw[e_idx, 0], F_raw[e_idx, 1]),
        title=f"Pareto front — {alt_id} (Nopt={N_SCEN_OPT})"
    )
    print(f"✅ Saved Pareto plot: {png} and {pdf}")

    runinfo = _build_runinfo_row(alt_id=alt_id, runtime_s=(t1 - t0), pareto_size=int(X.shape[0]))
    runinfo["ParetoFile"] = pareto_path
    runinfo["PlotPNG"] = png
    runinfo["PlotPDF"] = pdf

    return val_rows, runinfo


def save_batch_summary(batch_rows: List[dict], runinfo_rows: List[dict], stamp_batch: str):
    summary_path = os.path.join(RESULTS_DIR, f"NSGA2_CONVERGENCE_BATCH_SUMMARY_N20_{stamp_batch}.xlsx")
    df_sum = pd.DataFrame(batch_rows)
    df_run = pd.DataFrame(runinfo_rows)

    keep_cols = [
        "AltID", "ScenarioCount", "Selection",
        "Import_kWh_mean", "TotalCost_$yr_mean",
        "Vmin_pu_worst", "Vmax_pu_worst", "Export_kWh_worst", "ViolBH_worst",
        "SOC_min_worst", "SOC_max_worst",
        "val_robust_ok_v", "val_robust_ok_export", "val_robust_ok_soc",
        "val_robust_ok_violbh", "val_robust_ok_all",
        "Runtime_s", "Pareto_size", "ValidationFile"
    ]
    df_compact = df_sum[[c for c in keep_cols if c in df_sum.columns]].copy()

    with pd.ExcelWriter(summary_path, engine="openpyxl") as w:
        df_sum.to_excel(w, sheet_name="ValidationSummary_All", index=False)
        df_compact.to_excel(w, sheet_name="ValidationSummary_Compact", index=False)
        df_run.to_excel(w, sheet_name="RunInfo", index=False)

    print(f"\n✅ Saved batch summary workbook: {summary_path}")


def main():
    ensure_dir(RESULTS_DIR)
    stamp_batch = timestamp()

    sim = Simulator(
        line_xlsx=PATH_LINES_XLSX,
        load_xlsx=PATH_LOAD_XLSX,
        pv_xlsx=PATH_PV_XLSX,
        sheet_pload=SHEET_PLOAD,
        sheet_qload=SHEET_QLOAD,
        sheet_pv=SHEET_PV,
        pv_per_kw_col=PV_PER_KW_COL,
    )

    all_rows = []
    all_runinfo = []

    for alt_id, base_arch in ALT_BASE_ARCHS.items():
        val_rows, runinfo = run_alt(sim, alt_id, base_arch, stamp_batch=stamp_batch)
        all_rows.extend(val_rows)
        all_runinfo.append(runinfo)

    save_batch_summary(all_rows, all_runinfo, stamp_batch=stamp_batch)

    print("\nWrapper confirm: FC dispatchable with min frac =", FC_PMIN_FRAC)
    print(f"Optimization scenarios: N={N_SCEN_OPT}, seed={SEED_OPT}")
    print(f"Validation scenarios:   N={N_SCEN_VAL}, seed={SEED_VAL}")
    print("\n✅ N=20 convergence batch finished.")


if __name__ == "__main__":
    main()