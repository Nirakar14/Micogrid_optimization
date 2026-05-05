

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List

from simulator_wrapper import (
    Simulator, ScenarioConfig, CostConfig,
    arch_from_vector, compute_total_cost_annual_from_energies,
    bi, bfs_solve,
    dispatch_bess_basic, dispatch_microturbine_load_follow,
    dispatch_fuelcell_turn_down_for_noexport,
    extra_charge_bess_for_noexport, curtail_renewables,
    Vmin_lim, Vmax_lim, SOC_MIN, SOC_MAX,
    Sbase_kVA, SLACK_BUS, ENFORCE_NO_EXPORT,
)

PATH_LINES_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\newlineparameters.xlsx"

PATH_LOAD_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\CIGRE_15day_Loads_R1toR18_CLEAN_S95cap.xlsx"
SHEET_PLOAD     = "Active_kW_R1toR18"
SHEET_QLOAD     = "Reactive_kVAr_R1toR18"

PATH_PV_XLSX    = r"C:\Users\Dell\Desktop\GRT\Codes\PV_15day_Hourly_Profile_Khumaltar_NOCT.xlsx"
SHEET_PV        = "PV_Hourly_15days"
PV_PER_KW_COL   = "PV_kW_per_1kW_STC"


A1_PARETO_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\Stage2_NSAGII\Results\n20_convergence\NSGA2_SOFT_A1_Pareto_N20_20260402_143632.xlsx"
A2_PARETO_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\Stage2_NSAGII\Results\n20_convergence\NSGA2_SOFT_A2_Pareto_N20_20260402_143632.xlsx"

RESULTS_DIR = r"C:\Users\Dell\Desktop\GRT\Codes\Stage2_NSAGII\Results\repair_test_a1_a2"


N_SCEN_VAL = 20
SEED_VAL = 77


EXPORT_TOL_KWH = 1e-3
FC_PMIN_FRAC = 0.0


Q_SUPPORT_RATIO = 0.20   
MT_UPSIZE_FRAC = 0.01    


ALT_BASE_ARCHS = {
    "A1": {"PV16_kW": 10.0, "PV17_kW": 3.0, "WT16_kW": 0.0,  "BESS_P_kW": 0.0, "BESS_E_kWh": 0.0, "MT_kW": 30.0, "FC_kW": 0.0},
    "A2": {"PV16_kW": 10.0, "PV17_kW": 3.0, "WT16_kW": 10.0, "BESS_P_kW": 0.0, "BESS_E_kWh": 0.0, "MT_kW": 30.0, "FC_kW": 0.0},
}

REPAIR_VARIANTS = [
    "BASELINE",
    "QSUPPORT_ONLY",
    "MT_PLUS_10_ONLY",
    "QSUPPORT_AND_MT_PLUS_10",
]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def save_validation_excel(path: str, alt_id: str, sol_name: str, variant: str, df_per: pd.DataFrame, met: dict):
    df = df_per.copy()

    
    if "AltID" in df.columns:
        df = df.drop(columns=["AltID"])
    if "Solution" in df.columns:
        df = df.drop(columns=["Solution"])
    if "Variant" in df.columns:
        df = df.drop(columns=["Variant"])

    df.insert(0, "AltID", alt_id)
    df.insert(1, "Solution", sol_name)
    df.insert(2, "Variant", variant)

    summary = pd.DataFrame([met]).copy()


    for c in ["AltID", "Solution", "Variant"]:
        if c in summary.columns:
            summary = summary.drop(columns=[c])

    summary.insert(0, "AltID", alt_id)
    summary.insert(1, "Solution", sol_name)
    summary.insert(2, "Variant", variant)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="PerScenario", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)


def read_selected_vectors_from_pareto(path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(path, sheet_name="Pareto")

    req = ["s_PV16", "s_PV17", "s_WT16", "s_BESS_P", "s_BESS_E", "s_MT", "s_FC", "pick_knee", "pick_eps"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Pareto file missing columns: {missing}")

    vec_cols = ["s_PV16", "s_PV17", "s_WT16", "s_BESS_P", "s_BESS_E", "s_MT", "s_FC"]

    out = {}
    knee_rows = df[df["pick_knee"] == 1]
    eps_rows  = df[df["pick_eps"] == 1]

    if len(knee_rows) == 0:
        raise ValueError(f"No pick_knee=1 row found in {path}")
    if len(eps_rows) == 0:
        raise ValueError(f"No pick_eps=1 row found in {path}")

    out["KNEE"] = knee_rows.iloc[0][vec_cols].to_numpy(dtype=float)
    out["EPS"]  = eps_rows.iloc[0][vec_cols].to_numpy(dtype=float)
    return out


def tweak_base_arch(base_arch: Dict[str, float], variant: str) -> Dict[str, float]:
    arch = dict(base_arch)
    if variant in ("MT_PLUS_10_ONLY", "QSUPPORT_AND_MT_PLUS_10"):
        arch["MT_kW"] = arch["MT_kW"] * (1.0 + MT_UPSIZE_FRAC)
    return arch


def q_support_enabled(variant: str) -> bool:
    return variant in ("QSUPPORT_ONLY", "QSUPPORT_AND_MT_PLUS_10")


def pv_q_support_kvar(Ppv_kW: float, Prated_kW: float, enable: bool) -> float:
    if (not enable) or (Ppv_kW <= 1e-9) or (Prated_kW <= 1e-9):
        return 0.0
    return Q_SUPPORT_RATIO * min(Ppv_kW, Prated_kW)


class RepairSimulator(Simulator):
    def run_one_scenario_repair(
        self,
        arch: Dict[str, float],
        s: int,
        cfg: ScenarioConfig,
        q_support_on: bool,
        export_tol_kwh: float = 0.0,
        fc_pmin_frac: float = 0.0,
    ) -> Dict[str, float]:
        load_mult, pv_mult, wt_mult = self.get_multipliers(cfg)

        P_s  = self.P_base * load_mult[s, :, None]
        Q_s  = self.Q_base * load_mult[s, :, None]
        pv_s = self.pv_base * pv_mult[s, :]
        wt_s = self.wt_base * wt_mult[s, :]

        PV16 = float(arch["PV16_kW"])
        PV17 = float(arch["PV17_kW"])
        WT16 = float(arch["WT16_kW"])
        PB   = float(arch["BESS_P_kW"])
        EB   = float(arch["BESS_E_kWh"])
        PMT  = float(arch["MT_kW"])
        PFC  = float(arch["FC_kW"])

        soc = 0.50
        soc_min_t = soc
        soc_max_t = soc

        total_loss_kWh = 0.0
        minV_global, maxV_global = 999.0, 0.0
        viol_bh = 0.0

        E_import = 0.0
        E_export = 0.0

        E_pv_avail = 0.0
        E_wt_avail = 0.0
        E_ren_used = 0.0
        E_curt = 0.0
        E_mt = 0.0
        bess_throughput = 0.0
        E_fc = 0.0
        fc_curtail_kWh = 0.0
        E_q_support = 0.0

        Pfc18_set = PFC

        for t in range(360):
            hod = t % 24
            Pload_bus = P_s[t, :]
            Qload_bus = Q_s[t, :]

            Ppv16 = PV16 * pv_s[t]
            Ppv17 = PV17 * pv_s[t]
            Pwt16 = WT16 * wt_s[t]

            Pgen = np.zeros(18, dtype=float)
            Qgen = np.zeros(18, dtype=float)

       
            Pgen[bi("R16")] += (Ppv16 + Pwt16)
            Pgen[bi("R17")] += Ppv17

          
            Qpv16 = pv_q_support_kvar(Ppv16, PV16, q_support_on)
            Qpv17 = pv_q_support_kvar(Ppv17, PV17, q_support_on)
            Qgen[bi("R16")] += Qpv16
            Qgen[bi("R17")] += Qpv17
            E_q_support += (Qpv16 + Qpv17)

    
            Pfc18 = np.clip(Pfc18_set, fc_pmin_frac * PFC, PFC) if PFC > 0 else 0.0
            Pgen[bi("R18")] += Pfc18

            E_pv_avail += max(Ppv16, 0.0) + max(Ppv17, 0.0)
            E_wt_avail += max(Pwt16, 0.0)

            Pload_total = float(np.sum(Pload_bus))
            Pnet_prebess = Pload_total - float(np.sum(Pgen))

      
            Pbess4, soc = dispatch_bess_basic(hod, Pnet_prebess, soc, PB, EB)
            Pgen[bi("R4")] += Pbess4
            bess_throughput += abs(Pbess4)
            soc_min_t = min(soc_min_t, soc)
            soc_max_t = max(soc_max_t, soc)

     
            Pnet_preMT = Pload_total - float(np.sum(Pgen))
            Pmt15 = dispatch_microturbine_load_follow(Pnet_preMT, PMT, pmin_frac=0.30)
            Pgen[bi("R15")] += Pmt15
            E_mt += max(Pmt15, 0.0)

     
            Pnet_total = Pload_total - float(np.sum(Pgen))
            Pgrid_import = max(Pnet_total, 0.0)
            Pgrid_export = max(-Pnet_total, 0.0)

            
            if ENFORCE_NO_EXPORT and Pgrid_export > 1e-9:
                if PFC > 1e-9:
                    Pfc_new, fc_reduced = dispatch_fuelcell_turn_down_for_noexport(
                        Pfc_set_kW=Pfc18,
                        Pexport_kW=Pgrid_export,
                        Pfc_max_kW=PFC,
                        pmin_frac=fc_pmin_frac
                    )
                    if fc_reduced > 0:
                        Pgen[bi("R18")] -= fc_reduced
                        fc_curtail_kWh += fc_reduced
                        Pfc18 = Pfc_new
                        Pfc18_set = Pfc_new
                    Pnet_total = Pload_total - float(np.sum(Pgen))
                    Pgrid_export = max(-Pnet_total, 0.0)

                if Pgrid_export > 1e-9:
                    extra_ch, soc2 = extra_charge_bess_for_noexport(Pgrid_export, soc, PB, EB)
                    if extra_ch > 0:
                        Pgen[bi("R4")] += (-extra_ch)
                        soc = soc2
                        bess_throughput += abs(extra_ch)

                    soc_min_t = min(soc_min_t, soc)
                    soc_max_t = max(soc_max_t, soc)
                    Pnet_total = Pload_total - float(np.sum(Pgen))
                    Pgrid_export = max(-Pnet_total, 0.0)

                if Pgrid_export > 1e-9:
                    c1, c2, c3, Ppv16_new, Ppv17_new, Pwt16_new = curtail_renewables(Pgrid_export, Ppv16, Ppv17, Pwt16)
                    E_curt += (max(c1, 0.0) + max(c2, 0.0) + max(c3, 0.0))
                    Ppv16, Ppv17, Pwt16 = Ppv16_new, Ppv17_new, Pwt16_new

                    
                    Pgen[bi("R16")] = 0.0
                    Pgen[bi("R17")] = 0.0
                    Pgen[bi("R16")] += (Ppv16 + Pwt16)
                    Pgen[bi("R17")] += Ppv17

                  
                    Qgen[bi("R16")] = pv_q_support_kvar(Ppv16, PV16, q_support_on)
                    Qgen[bi("R17")] = pv_q_support_kvar(Ppv17, PV17, q_support_on)

                    Pnet_total = Pload_total - float(np.sum(Pgen))
                    Pgrid_export = max(-Pnet_total, 0.0)

            Pgrid_import = max(Pnet_total, 0.0)
            Pgrid_export = max(-Pnet_total, 0.0)

            E_ren_used += max(Ppv16, 0.0) + max(Ppv17, 0.0) + max(Pwt16, 0.0)
            E_fc += max(Pfc18, 0.0)
            E_import += Pgrid_import
            E_export += max(Pgrid_export, 0.0)

            Pnet_bus = Pload_bus - Pgen
            Qnet_bus = Qload_bus - Qgen

            Sbus_pu = {}
            for i in range(18):
                bus = f"R{i+1}"
                if bus == SLACK_BUS:
                    continue
                S_kVA = (Pnet_bus[i] + 1j * Qnet_bus[i])
                Sbus_pu[bus] = S_kVA / Sbase_kVA

            V, Ploss_pu = bfs_solve(Sbus_pu, self.buses, self.parent, self.children, self.z, self.post, slack=SLACK_BUS)
            total_loss_kWh += (Ploss_pu * Sbase_kVA)

            Vmag = np.array([abs(V[b]) for b in self.buses], dtype=float)
            vmin = float(Vmag.min())
            vmax = float(Vmag.max())
            minV_global = min(minV_global, vmin)
            maxV_global = max(maxV_global, vmax)
            viol_bh += int(np.sum((Vmag < Vmin_lim) | (Vmag > Vmax_lim)))

        E_ren_avail = E_pv_avail + E_wt_avail
        ren_util = 0.0 if E_ren_avail <= 1e-9 else (100.0 * E_ren_used / E_ren_avail)
        bess_cycles = (bess_throughput / (2.0 * EB)) if EB > 1e-9 else 0.0

        return {
            "Import_kWh": float(E_import),
            "Export_kWh": float(E_export),
            "Vmin_pu": float(minV_global),
            "Vmax_pu": float(maxV_global),
            "ViolBH": float(viol_bh),
            "Loss_kWh": float(total_loss_kWh),
            "RenAvail_kWh": float(E_ren_avail),
            "RenUsed_kWh": float(E_ren_used),
            "RenUtil_pct": float(np.clip(ren_util, 0.0, 100.0)),
            "Curt_kWh": float(max(E_curt, 0.0)),
            "MT_kWh": float(E_mt),
            "FC_kWh": float(E_fc),
            "FC_Curt_kWh": float(fc_curtail_kWh),
            "BESS_throughput_kWh": float(bess_throughput),
            "BESS_cycles": float(bess_cycles),
            "SOC_end": float(soc),
            "SOC_min": float(soc_min_t),
            "SOC_max": float(soc_max_t),
            "QSupport_kVArh": float(E_q_support),
        }

    def evaluate_architecture_repair(
        self,
        x: np.ndarray,
        base_arch: Dict[str, float],
        scenario_cfg: ScenarioConfig,
        cost_cfg: CostConfig,
        variant: str,
        arch_id: str,
        export_tol_kwh: float = 0.0,
        fc_pmin_frac: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
        base_arch_mod = tweak_base_arch(base_arch, variant)
        arch = arch_from_vector(x, base_arch_mod)

        q_on = q_support_enabled(variant)

        per = []
        for s in range(scenario_cfg.N):
            if variant == "BASELINE":
                out = self.run_one_scenario(
                    arch, s, scenario_cfg,
                    export_tol_kwh=export_tol_kwh,
                    fc_pmin_frac=fc_pmin_frac
                )
            else:
                out = self.run_one_scenario_repair(
                    arch, s, scenario_cfg,
                    q_support_on=q_on,
                    export_tol_kwh=export_tol_kwh,
                    fc_pmin_frac=fc_pmin_frac
                )
            per.append(out)

        df = pd.DataFrame(per)
        T = int(self.P_base.shape[0])

        costs = []
        for _, r in df.iterrows():
            c, _ = compute_total_cost_annual_from_energies(
                arch,
                import_kwh=float(r["Import_kWh"]),
                mt_kwh=float(r["MT_kWh"]),
                thr_kwh=float(r["BESS_throughput_kWh"]),
                cost_cfg=cost_cfg,
                horizon_hours=T
            )
            costs.append(c)
        df["TotalCost_$yr"] = np.asarray(costs, dtype=float)

        def _mean_std_worst(col: str, worst_is: str = "max"):
            v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(float)
            mean = float(np.mean(v))
            std  = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            worst = float(np.max(v) if worst_is == "max" else np.min(v))
            return mean, std, worst

        met = {"Arch_ID": arch_id, "Variant": variant}

        for col, worst_is in [
            ("Import_kWh", "max"),
            ("Export_kWh", "max"),
            ("Vmin_pu", "min"),
            ("Vmax_pu", "max"),
            ("ViolBH", "max"),
            ("Loss_kWh", "max"),
            ("Curt_kWh", "max"),
            ("RenUtil_pct", "min"),
            ("BESS_cycles", "max"),
            ("BESS_throughput_kWh", "max"),
            ("MT_kWh", "max"),
            ("FC_kWh", "max"),
            ("FC_Curt_kWh", "max"),
            ("SOC_end", "min"),
            ("SOC_min", "min"),
            ("SOC_max", "max"),
            ("TotalCost_$yr", "max"),
            ("QSupport_kVArh", "max"),
        ]:
            if col in df.columns:
                m, s, w = _mean_std_worst(col, worst_is=worst_is)
                met[f"{col}_mean"] = m
                met[f"{col}_std"] = s
                met[f"{col}_worst"] = w

        total_cost_mean, cost_break = compute_total_cost_annual_from_energies(
            arch,
            import_kwh=float(met["Import_kWh_mean"]),
            mt_kwh=float(met["MT_kWh_mean"]),
            thr_kwh=float(met["BESS_throughput_kWh_mean"]),
            cost_cfg=cost_cfg,
            horizon_hours=T
        )
        met.update(cost_break)

        obj = {
            "Import_kWh_mean": float(met["Import_kWh_mean"]),
            "TotalCost_$yr_mean": float(total_cost_mean),
        }
        return obj, met, df


def build_case_list() -> List[Tuple[str, str, str]]:
    return [
        ("A1", "KNEE", A1_PARETO_XLSX),
        ("A1", "EPS",  A1_PARETO_XLSX),
        ("A2", "KNEE", A2_PARETO_XLSX),
        ("A2", "EPS",  A2_PARETO_XLSX),
    ]


def main():
    ensure_dir(RESULTS_DIR)
    stamp = timestamp()

    sim = RepairSimulator(
        line_xlsx=PATH_LINES_XLSX,
        load_xlsx=PATH_LOAD_XLSX,
        pv_xlsx=PATH_PV_XLSX,
        sheet_pload=SHEET_PLOAD,
        sheet_qload=SHEET_QLOAD,
        sheet_pv=SHEET_PV,
        pv_per_kw_col=PV_PER_KW_COL,
    )

    scen_val = ScenarioConfig(N=N_SCEN_VAL, seed=SEED_VAL)
    cost_cfg = CostConfig()

    vectors = {
        "A1": read_selected_vectors_from_pareto(A1_PARETO_XLSX),
        "A2": read_selected_vectors_from_pareto(A2_PARETO_XLSX),
    }

    rows = []

    for alt_id, sol_name, _ in build_case_list():
        x = vectors[alt_id][sol_name]
        base_arch = ALT_BASE_ARCHS[alt_id]

        for variant in REPAIR_VARIANTS:
            obj, met, df_per = sim.evaluate_architecture_repair(
                x=x,
                base_arch=base_arch,
                scenario_cfg=scen_val,
                cost_cfg=cost_cfg,
                variant=variant,
                arch_id=f"{alt_id}_{sol_name}_{variant}",
                export_tol_kwh=EXPORT_TOL_KWH,
                fc_pmin_frac=FC_PMIN_FRAC,
            )
            ok = robust_ok_from_metrics(met)
            met.update(ok)

            out_path = os.path.join(
                RESULTS_DIR,
                f"REPAIR_{alt_id}_{sol_name}_{variant}_{stamp}.xlsx"
            )
            save_validation_excel(out_path, alt_id, sol_name, variant, df_per, met)

            row = {
                "AltID": alt_id,
                "Solution": sol_name,
                "Variant": variant,
                "x_sPV16": x[0],
                "x_sPV17": x[1],
                "x_sWT16": x[2],
                "x_sBESSP": x[3],
                "x_sBESSE": x[4],
                "x_sMT": x[5],
                "x_sFC": x[6],
                "Import_kWh_mean": obj["Import_kWh_mean"],
                "TotalCost_$yr_mean": obj["TotalCost_$yr_mean"],
                "Vmin_pu_worst": met.get("Vmin_pu_worst", np.nan),
                "Vmax_pu_worst": met.get("Vmax_pu_worst", np.nan),
                "Export_kWh_worst": met.get("Export_kWh_worst", np.nan),
                "ViolBH_worst": met.get("ViolBH_worst", np.nan),
                "SOC_min_worst": met.get("SOC_min_worst", np.nan),
                "SOC_max_worst": met.get("SOC_max_worst", np.nan),
                "QSupport_kVArh_mean": met.get("QSupport_kVArh_mean", 0.0),
                "val_robust_ok_v": ok["val_robust_ok_v"],
                "val_robust_ok_export": ok["val_robust_ok_export"],
                "val_robust_ok_soc": ok["val_robust_ok_soc"],
                "val_robust_ok_violbh": ok["val_robust_ok_violbh"],
                "val_robust_ok_all": ok["val_robust_ok_all"],
                "ValidationFile": out_path,
            }
            rows.append(row)

            print(f"\n{alt_id} | {sol_name} | {variant}")
            print(f"Import={row['Import_kWh_mean']:.3f}, Cost={row['TotalCost_$yr_mean']:.3f}, "
                  f"Vmin_worst={row['Vmin_pu_worst']:.4f}, ViolBH_worst={row['ViolBH_worst']:.0f}, "
                  f"Robust={row['val_robust_ok_all']}")

    df = pd.DataFrame(rows)
    summary_path = os.path.join(RESULTS_DIR, f"REPAIR_TEST_SUMMARY_A1_A2_{stamp}.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="RepairSummary", index=False)

    print(f"\n✅ Saved repair summary: {summary_path}")


if __name__ == "__main__":
    main()