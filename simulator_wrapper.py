
from __future__ import annotations

import os
import atexit
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
import multiprocessing as mp
from typing import Dict, Tuple, Any, Optional, List

DEFAULT_LINE_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\newlineparameters.xlsx"

DEFAULT_LOAD_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\CIGRE_15day_Loads_R1toR18_CLEAN_S95cap.xlsx"
DEFAULT_SHEET_PLOAD = "Active_kW_R1toR18"
DEFAULT_SHEET_QLOAD = "Reactive_kVAr_R1toR18"

DEFAULT_PV_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\PV_15day_Hourly_Profile_Khumaltar_NOCT.xlsx"
DEFAULT_SHEET_PV = "PV_Hourly_15days"
DEFAULT_PV_PER_KW_COL = "PV_kW_per_1kW_STC"


VLL_kV    = 0.4
Sbase_kVA = 400.0
SLACK_BUS = "R1"
Vslack_pu = 1.0 + 0j

ALL_BUSES = [f"R{i}" for i in range(1, 19)]


Vmin_lim = 0.95
Vmax_lim = 1.05

ENFORCE_NO_EXPORT = True  


SOC0    = 0.50
SOC_MIN = 0.20
SOC_MAX = 0.90
ETA_CH  = 0.95
ETA_DIS = 0.95
PEAK_HOURS = {18, 19, 20, 21}


WIND_SEED     = 123
WIND_MEAN_MS  = 4.5
WIND_K_SHAPE  = 2.0
V_CI, V_R, V_CO = 3.0, 12.0, 25.0

=
FC_PMIN_FRAC_DEFAULT = 0.0 



class ScenarioConfig:
 
    def __init__(
        self,
        N: int = 10,
        seed: int = 7,
        load_daily_sigma: float = 0.06,
        load_hourly_sigma: float = 0.03,
        load_clip: Tuple[float, float] = (0.85, 1.20),
        pv_daily_sigma: float = 0.12,
        pv_hourly_sigma: float = 0.06,
        pv_clip: Tuple[float, float] = (0.0, 1.25),
        wt_daily_sigma: float = 0.15,
        wt_hourly_sigma: float = 0.10,
        wt_clip: Tuple[float, float] = (0.0, 1.40),
        enable_pv_dips: bool = True,
        pv_dip_prob_per_day: float = 0.15,
        pv_dip_hours: Tuple[int, int] = (11, 15),
        pv_dip_strength: Tuple[float, float] = (0.60, 0.85),
        # alias
        n_scen: Optional[int] = None,
    ):
        if n_scen is not None:
            N = int(n_scen)
        self.N = int(N)
        self.seed = int(seed)

        self.load_daily_sigma = float(load_daily_sigma)
        self.load_hourly_sigma = float(load_hourly_sigma)
        self.load_clip = tuple(load_clip)

        self.pv_daily_sigma = float(pv_daily_sigma)
        self.pv_hourly_sigma = float(pv_hourly_sigma)
        self.pv_clip = tuple(pv_clip)

        self.wt_daily_sigma = float(wt_daily_sigma)
        self.wt_hourly_sigma = float(wt_hourly_sigma)
        self.wt_clip = tuple(wt_clip)

        self.enable_pv_dips = bool(enable_pv_dips)
        self.pv_dip_prob_per_day = float(pv_dip_prob_per_day)
        self.pv_dip_hours = tuple(pv_dip_hours)
        self.pv_dip_strength = tuple(pv_dip_strength)

    def __repr__(self) -> str:
        return f"ScenarioConfig(N={self.N}, seed={self.seed})"

    def cache_key(self) -> Tuple:
        return (
            self.N, self.seed,
            self.load_daily_sigma, self.load_hourly_sigma, self.load_clip,
            self.pv_daily_sigma, self.pv_hourly_sigma, self.pv_clip,
            self.wt_daily_sigma, self.wt_hourly_sigma, self.wt_clip,
            self.enable_pv_dips, self.pv_dip_prob_per_day, self.pv_dip_hours, self.pv_dip_strength
        )



@dataclass(frozen=True)
class CostConfig:
    
    pv_capex_per_kw: float = 1340.0
    wt_capex_per_kw: float = 3036.0
    bess_capex_per_kw: float = 525.0
    bess_capex_per_kwh: float = 445.0
    mt_capex_per_kw: float = 3134.0
    fc_capex_per_kw: float = 7150.0

    pv_fom_per_kwyr: float = 18.03
    wt_fom_per_kwyr: float = 38.04
    mt_fom_per_kwyr: float = 0.0
    fc_fom_per_kwyr: float = 0.0
    bess_fom_per_kwyr: float = 0.0

    grid_cost_per_kwh: float = 0.1241
    mt_fuel_cost_per_kwh: float = 0.0732

    
    batt_deg_cost_per_kwh_throughput: float = 0.0813


    discount_rate: float = 0.08
    lifetime_years: int = 20

  
    annualize_from_horizon: bool = True  


def capital_recovery_factor(r: float, n: int) -> float:
    r = float(r)
    n = int(n)
    if n <= 0:
        return 1.0
    if abs(r) < 1e-12:
        return 1.0 / n
    a = (1.0 + r) ** n
    return (r * a) / (a - 1.0)



def normalize_bus_name(x) -> str:
    s = str(x).strip()
    if not s:
        return s
    s = s.replace(" ", "")
    if s.upper().startswith("R"):
        return "R" + s[1:].strip()
    try:
        n = int(float(s))
        return f"R{n}"
    except Exception:
        return s


def bi(busname: str) -> int:
    return int(busname[1:]) - 1


def read_lines(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    if raw.shape[1] < 4:
        raise ValueError(f"Line file must have at least 4 columns. Found shape {raw.shape}")
    raw = raw.iloc[:, :4].copy()

    first_row = raw.iloc[0].astype(str).str.strip().str.lower().tolist()
    header_like = (
        ("from" in first_row[0] and "to" in first_row[1]) or
        ("from_bus" in first_row[0]) or
        ("to_bus" in first_row[1])
    )
    if header_like:
        raw = raw.iloc[1:, :].copy()

    raw.columns = ["from_bus", "to_bus", "R", "X"]
    raw["from_bus"] = raw["from_bus"].apply(normalize_bus_name)
    raw["to_bus"]   = raw["to_bus"].apply(normalize_bus_name)
    raw["R"] = pd.to_numeric(raw["R"], errors="coerce")
    raw["X"] = pd.to_numeric(raw["X"], errors="coerce")
    raw = raw.dropna(subset=["from_bus", "to_bus", "R", "X"]).reset_index(drop=True)
    return raw


def build_tree_strict(lines: pd.DataFrame, slack: str = "R1"):
    buses = sorted(set(lines["from_bus"]).union(set(lines["to_bus"])).union(set(ALL_BUSES)))
    if slack not in buses:
        raise ValueError(f"Slack '{slack}' not in buses parsed from line sheet.")

    adj = {b: [] for b in buses}
    z_und = {}
    for _, row in lines.iterrows():
        a, b = row["from_bus"], row["to_bus"]
        Z = row["Zpu"]
        adj[a].append(b)
        adj[b].append(a)
        z_und[(a, b)] = Z
        z_und[(b, a)] = Z

    parent = {b: None for b in buses}
    seen = set([slack])
    q = [slack]
    while q:
        u = q.pop(0)
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                parent[v] = u
                q.append(v)

    missing = [b for b in ALL_BUSES if b not in seen]
    if missing:
        raise ValueError(f"Network not connected from {slack}. Unreachable buses: {missing}")

    children = {b: [] for b in buses}
    z = {}
    for b in buses:
        p = parent[b]
        if p is not None:
            children[p].append(b)
            z[(p, b)] = z_und[(p, b)]

    post = []
    def dfs(u):
        for v in children.get(u, []):
            dfs(v)
        post.append(u)
    dfs(slack)

    return buses, parent, children, z, post


def read_bus_matrix_from_sheet(path: str, sheet: str) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=sheet)
    cols = [c for c in df.columns if str(c).strip().upper().startswith("R")]
    if not cols:
        raise ValueError(f"Sheet '{sheet}' has no R* columns.")
    cols_norm = [normalize_bus_name(c) for c in cols]
    df = df[cols].copy()
    df.columns = cols_norm
    if len(df) < 360:
        raise ValueError(f"Sheet '{sheet}' has {len(df)} rows; need >=360.")
    df = df.iloc[:360].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for b in ALL_BUSES:
        if b not in df.columns:
            df[b] = 0.0
    return df[ALL_BUSES].values.astype(float)


def read_pv_per_kw_profile(path: str, sheet: str, col: str) -> np.ndarray:
    df = pd.read_excel(path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"PV sheet '{sheet}' missing '{col}'.")
    pv = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
    if len(pv) < 360:
        raise ValueError(f"PV profile length {len(pv)} < 360.")
    return np.clip(pv[:360].astype(float), 0.0, 1.25)


def weibull_wind_speed(n, mean, k, rng):
    from math import gamma as Gamma
    lam = mean / Gamma(1.0 + 1.0/k)
    return rng.weibull(k, size=n) * lam


def diurnal_wind_factor(hour):
    return 1.0 + 0.10*np.sin((hour - 14) * 2*np.pi/24)


def wind_power_cf(v, v_ci=V_CI, v_r=V_R, v_co=V_CO):
    cf = np.zeros_like(v, dtype=float)
    m1 = (v >= v_ci) & (v < v_r)
    m2 = (v >= v_r) & (v <= v_co)
    cf[m1] = ((v[m1] - v_ci) / (v_r - v_ci))**3
    cf[m2] = 1.0
    return np.clip(cf, 0.0, 1.0)


def build_wind_cf_360():
    rng = np.random.default_rng(WIND_SEED)
    v = weibull_wind_speed(360, WIND_MEAN_MS, WIND_K_SHAPE, rng)
    hours = np.arange(360) % 24
    v = v * np.array([diurnal_wind_factor(h) for h in hours])
    return wind_power_cf(v).astype(float)



def bfs_solve(Sbus_pu: Dict[str, complex], buses, parent, children, z, post,
              slack: str = "R1", max_iter: int = 80, tol: float = 1e-8):
    V = {b: (Vslack_pu if b == slack else 1.0 + 0j) for b in buses}
    Iinj = {b: 0j for b in buses}
    Ibr  = {(f, t): 0j for (f, t) in z.keys()}

    for _ in range(max_iter):
        Vprev = V.copy()

        for b in buses:
            if b == slack:
                Iinj[b] = 0j
                continue
            S = Sbus_pu.get(b, 0j)
            Iinj[b] = 0j if abs(V[b]) < 1e-12 else np.conj(S / V[b])

        Iacc = {b: Iinj[b] for b in buses}
        for b in post:
            p = parent.get(b, None)
            if p is not None and (p, b) in Ibr:
                Ibr[(p, b)] = Iacc[b]
                Iacc[p] += Iacc[b]

        V[slack] = Vslack_pu
        stack = [slack]
        while stack:
            u = stack.pop()
            for v in children.get(u, []):
                V[v] = V[u] - z[(u, v)] * Ibr[(u, v)]
                stack.append(v)

        dv = max(abs(V[b] - Vprev[b]) for b in buses)
        if dv < tol:
            break

    Ploss_pu = 0.0
    for (f, t), I in Ibr.items():
        Ploss_pu += (abs(I)**2) * z[(f, t)].real

    return V, Ploss_pu



def dispatch_bess_basic(hour_of_day: int, Pnet_prebess_kW: float, soc: float,
                       Pmax_kW: float, E_kWh: float):
    if Pmax_kW <= 1e-9 or E_kWh <= 1e-9:
        return 0.0, soc

   
    if hour_of_day in PEAK_HOURS and Pnet_prebess_kW > 0 and soc > SOC_MIN + 1e-6:
        E_avail = (soc - SOC_MIN) * E_kWh
        P_energy_limit = E_avail * ETA_DIS
        Pdis = min(Pmax_kW, Pnet_prebess_kW, P_energy_limit)
        soc_next = soc - (Pdis / ETA_DIS) / E_kWh
        return Pdis, max(SOC_MIN, soc_next)

 
    if 10 <= hour_of_day <= 15 and soc < SOC_MAX - 1e-6:
        E_room = (SOC_MAX - soc) * E_kWh
        P_energy_limit = E_room / ETA_CH
        Pch = min(Pmax_kW, P_energy_limit)
        soc_next = soc + (ETA_CH * Pch) / E_kWh
        return -Pch, min(SOC_MAX, soc_next)

    return 0.0, soc


def extra_charge_bess_for_noexport(Pexport_kW: float, soc: float, Pmax_kW: float, E_kWh: float):
    if Pexport_kW <= 1e-9 or Pmax_kW <= 1e-9 or E_kWh <= 1e-9:
        return 0.0, soc
    if soc >= SOC_MAX - 1e-6:
        return 0.0, soc

    E_room = (SOC_MAX - soc) * E_kWh
    P_energy_limit = E_room / ETA_CH
    Pch = min(Pmax_kW, Pexport_kW, P_energy_limit)
    soc_next = soc + (ETA_CH * Pch) / E_kWh
    return Pch, min(SOC_MAX, soc_next)


def dispatch_microturbine_load_follow(P_remaining_kW: float, Pmax_kW: float, pmin_frac: float = 0.30):
    if Pmax_kW <= 1e-9 or P_remaining_kW <= 0:
        return 0.0
    Pmin = float(pmin_frac) * Pmax_kW
    if P_remaining_kW < Pmin:
        return 0.0
    return min(P_remaining_kW, Pmax_kW)


def dispatch_fuelcell_turn_down_for_noexport(Pfc_set_kW: float, Pexport_kW: float,
                                           Pfc_max_kW: float, pmin_frac: float = FC_PMIN_FRAC_DEFAULT):
    
    if Pfc_max_kW <= 1e-9:
        return 0.0, 0.0

    Pmin = float(pmin_frac) * float(Pfc_max_kW)
    Pfc_set = float(np.clip(Pfc_set_kW, Pmin, Pfc_max_kW))

    if Pexport_kW <= 1e-9:
        return Pfc_set, 0.0

    
    reduce = min(Pexport_kW, max(0.0, Pfc_set - Pmin))
    Pfc_new = Pfc_set - reduce
    return Pfc_new, reduce


def curtail_renewables(Pneeded_kW: float, Ppv16: float, Ppv17: float, Pwt16: float):
    avail = max(Ppv16, 0) + max(Ppv17, 0) + max(Pwt16, 0)
    if Pneeded_kW <= 1e-9 or avail <= 1e-9:
        return 0.0, 0.0, 0.0, Ppv16, Ppv17, Pwt16

    x = min(Pneeded_kW, avail)
    a1 = max(Ppv16, 0) / avail
    a2 = max(Ppv17, 0) / avail
    a3 = max(Pwt16, 0) / avail

    c1 = a1 * x
    c2 = a2 * x
    c3 = a3 * x
    return c1, c2, c3, (Ppv16 - c1), (Ppv17 - c2), (Pwt16 - c3)



def _lognormal_mult(rng, sigma, size):
    x = rng.normal(0.0, sigma, size=size)
    return np.exp(x - 0.5 * sigma * sigma)


def make_scenario_multipliers(T: int, cfg: ScenarioConfig):
    assert T % 24 == 0
    days = T // 24
    rng = np.random.default_rng(cfg.seed)

    load_mult = np.zeros((cfg.N, T), dtype=float)
    pv_mult   = np.zeros((cfg.N, T), dtype=float)
    wt_mult   = np.zeros((cfg.N, T), dtype=float)

    for s in range(cfg.N):
        load_day = _lognormal_mult(rng, cfg.load_daily_sigma, size=days)
        pv_day   = _lognormal_mult(rng, cfg.pv_daily_sigma,   size=days)
        wt_day   = _lognormal_mult(rng, cfg.wt_daily_sigma,   size=days)

        load_hr = _lognormal_mult(rng, cfg.load_hourly_sigma, size=T)
        pv_hr   = _lognormal_mult(rng, cfg.pv_hourly_sigma,   size=T)
        wt_hr   = _lognormal_mult(rng, cfg.wt_hourly_sigma,   size=T)

        load_day_hr = np.repeat(load_day, 24)
        pv_day_hr   = np.repeat(pv_day, 24)
        wt_day_hr   = np.repeat(wt_day, 24)

        load_mult[s, :] = np.clip(load_day_hr * load_hr, cfg.load_clip[0], cfg.load_clip[1])
        pv_mult[s, :]   = np.clip(pv_day_hr   * pv_hr,   cfg.pv_clip[0],   cfg.pv_clip[1])
        wt_mult[s, :]   = np.clip(wt_day_hr   * wt_hr,   cfg.wt_clip[0],   cfg.wt_clip[1])

        if cfg.enable_pv_dips:
            for d in range(days):
                if rng.random() < cfg.pv_dip_prob_per_day:
                    h0, h1 = cfg.pv_dip_hours
                    dip = rng.uniform(cfg.pv_dip_strength[0], cfg.pv_dip_strength[1])
                    idx0 = d * 24 + h0
                    idx1 = d * 24 + h1 + 1
                    pv_mult[s, idx0:idx1] *= dip
            pv_mult[s, :] = np.clip(pv_mult[s, :], cfg.pv_clip[0], cfg.pv_clip[1])

    return load_mult, pv_mult, wt_mult



def arch_from_vector(x, base_arch: Dict[str, float]) -> Dict[str, float]:
    x = np.array(x, dtype=float).flatten()
    if x.size != 7:
        raise ValueError("x must have 7 scale variables.")
    s = np.clip(x, 0.0, 2.0)

    arch = dict(base_arch)
    arch["PV16_kW"]     = float(base_arch["PV16_kW"]     * s[0])
    arch["PV17_kW"]     = float(base_arch["PV17_kW"]     * s[1])
    arch["WT16_kW"]     = float(base_arch["WT16_kW"]     * s[2])
    arch["BESS_P_kW"]   = float(base_arch["BESS_P_kW"]   * s[3])
    arch["BESS_E_kWh"]  = float(base_arch["BESS_E_kWh"]  * s[4])
    arch["MT_kW"]       = float(base_arch["MT_kW"]       * s[5])
    arch["FC_kW"]       = float(base_arch["FC_kW"]       * s[6])
    return arch



def _cost_constants(arch: Dict[str, float], cost_cfg: CostConfig):
    capex = (
        cost_cfg.pv_capex_per_kw * (arch["PV16_kW"] + arch["PV17_kW"]) +
        cost_cfg.wt_capex_per_kw * arch["WT16_kW"] +
        cost_cfg.bess_capex_per_kw * arch["BESS_P_kW"] +
        cost_cfg.bess_capex_per_kwh * arch["BESS_E_kWh"] +
        cost_cfg.mt_capex_per_kw * arch["MT_kW"] +
        cost_cfg.fc_capex_per_kw * arch["FC_kW"]
    )

    crf = capital_recovery_factor(cost_cfg.discount_rate, cost_cfg.lifetime_years)
    capex_annual = crf * capex

    fom = (
        cost_cfg.pv_fom_per_kwyr * (arch["PV16_kW"] + arch["PV17_kW"]) +
        cost_cfg.wt_fom_per_kwyr * arch["WT16_kW"] +
        cost_cfg.mt_fom_per_kwyr * arch["MT_kW"] +
        cost_cfg.fc_fom_per_kwyr * arch["FC_kW"] +
        cost_cfg.bess_fom_per_kwyr * arch["BESS_P_kW"]
    )

    return float(capex), float(crf), float(capex_annual), float(fom)


def compute_total_cost_annual_from_energies(
    arch: Dict[str, float],
    import_kwh: float,
    mt_kwh: float,
    thr_kwh: float,
    cost_cfg: CostConfig,
    horizon_hours: int = 360
):
    capex, crf, capex_annual, fom = _cost_constants(arch, cost_cfg)

    sf = (8760.0 / float(horizon_hours)) if cost_cfg.annualize_from_horizon else 1.0
    grid_cost = cost_cfg.grid_cost_per_kwh * (float(import_kwh) * sf)
    fuel_cost = cost_cfg.mt_fuel_cost_per_kwh * (float(mt_kwh) * sf)
    deg_cost  = cost_cfg.batt_deg_cost_per_kwh_throughput * (float(thr_kwh) * sf)

    total = capex_annual + fom + grid_cost + fuel_cost + deg_cost

    breakdown = {
        "CAPEX_$": float(capex),
        "CRF": float(crf),
        "CAPEX_annual_$yr": float(capex_annual),
        "FixedOM_$yr": float(fom),
        "GridCost_$yr": float(grid_cost),
        "FuelCost_$yr": float(fuel_cost),
        "BattDegCost_$yr": float(deg_cost),
        "ScaleFactor_varcost": float(sf),
        "TotalCost_$yr": float(total),
    }
    return float(total), breakdown



def _simulate_one_scenario(
    arch: Dict[str, float],
    P_s: np.ndarray,
    Q_s: np.ndarray,
    pv_s: np.ndarray,
    wt_s: np.ndarray,
    parent, children, z, post,
    export_tol_kwh: float = 0.0,
    fc_pmin_frac: float = FC_PMIN_FRAC_DEFAULT
) -> Dict[str, float]:

    PV16 = float(arch["PV16_kW"])
    PV17 = float(arch["PV17_kW"])
    WT16 = float(arch["WT16_kW"])
    PB   = float(arch["BESS_P_kW"])
    EB   = float(arch["BESS_E_kWh"])
    PMT  = float(arch["MT_kW"])
    PFC_MAX  = float(arch["FC_kW"])  # nameplate max

    soc = SOC0
    soc_min_t = soc
    soc_max_t = soc

    total_loss_kWh = 0.0
    viol_bh = 0
    minV_global, maxV_global = 999.0, 0.0
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

    
    Pfc18_set = PFC_MAX

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

        
        Pfc18 = np.clip(Pfc18_set, fc_pmin_frac * PFC_MAX, PFC_MAX) if PFC_MAX > 0 else 0.0
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
            
            if PFC_MAX > 1e-9:
                Pfc_new, fc_reduced = dispatch_fuelcell_turn_down_for_noexport(
                    Pfc_set_kW=Pfc18,
                    Pexport_kW=Pgrid_export,
                    Pfc_max_kW=PFC_MAX,
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

                Pnet_total = Pload_total - float(np.sum(Pgen))
                Pgrid_export = max(-Pnet_total, 0.0)

        Pgrid_import = max(Pnet_total, 0.0)
        Pgrid_export = max(-Pnet_total, 0.0)

        
        ren_used_this = max(Ppv16, 0.0) + max(Ppv17, 0.0) + max(Pwt16, 0.0)
        E_ren_used += ren_used_this

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

        V, Ploss_pu = bfs_solve(Sbus_pu, ALL_BUSES, parent, children, z, post, slack=SLACK_BUS)
        total_loss_kWh += (Ploss_pu * Sbase_kVA)

        Vmag = np.array([abs(V[b]) for b in ALL_BUSES], dtype=float)
        vmin = float(Vmag.min())
        vmax = float(Vmag.max())
        minV_global = min(minV_global, vmin)
        maxV_global = max(maxV_global, vmax)
        viol_bh += int(np.sum((Vmag < Vmin_lim) | (Vmag > Vmax_lim)))

    E_ren_avail = E_pv_avail + E_wt_avail
    ren_util = 100.0 if E_ren_avail <= 1e-9 else (100.0 * E_ren_used / E_ren_avail)
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
    }



class Simulator:
    def __init__(
        self,
        line_xlsx: str = DEFAULT_LINE_XLSX,
        load_xlsx: str = DEFAULT_LOAD_XLSX,
        pv_xlsx: str = DEFAULT_PV_XLSX,
        sheet_pload: str = DEFAULT_SHEET_PLOAD,
        sheet_qload: str = DEFAULT_SHEET_QLOAD,
        sheet_pv: str = DEFAULT_SHEET_PV,
        pv_per_kw_col: str = DEFAULT_PV_PER_KW_COL,
    ):
        self.line_xlsx = line_xlsx
        self.load_xlsx = load_xlsx
        self.pv_xlsx = pv_xlsx
        self.sheet_pload = sheet_pload
        self.sheet_qload = sheet_qload
        self.sheet_pv = sheet_pv
        self.pv_per_kw_col = pv_per_kw_col

        lines = read_lines(self.line_xlsx)

        VLL = VLL_kV * 1e3
        Sbase = Sbase_kVA * 1e3
        Zbase = (VLL**2) / Sbase
        lines["Zpu"] = (lines["R"].values + 1j*lines["X"].values) / Zbase

        buses, parent, children, z, post = build_tree_strict(lines, slack=SLACK_BUS)
        self.buses = buses
        self.parent = parent
        self.children = children
        self.z = z
        self.post = post

        self.P_base = read_bus_matrix_from_sheet(self.load_xlsx, self.sheet_pload)
        self.Q_base = read_bus_matrix_from_sheet(self.load_xlsx, self.sheet_qload)

        self.pv_base = read_pv_per_kw_profile(self.pv_xlsx, self.sheet_pv, self.pv_per_kw_col)
        self.wt_base = build_wind_cf_360()

        self._mult_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_multipliers(self, cfg: ScenarioConfig):
        key = cfg.cache_key()
        if key not in self._mult_cache:
            T = self.P_base.shape[0]
            self._mult_cache[key] = make_scenario_multipliers(T, cfg)
        return self._mult_cache[key]

    def run_one_scenario(self, arch: Dict[str, float], s: int, cfg: ScenarioConfig,
                         export_tol_kwh: float = 0.0, fc_pmin_frac: float = FC_PMIN_FRAC_DEFAULT) -> Dict[str, float]:
        load_mult, pv_mult, wt_mult = self.get_multipliers(cfg)

        P_s  = self.P_base * load_mult[s, :, None]
        Q_s  = self.Q_base * load_mult[s, :, None]
        pv_s = self.pv_base * pv_mult[s, :]
        wt_s = self.wt_base * wt_mult[s, :]

        return _simulate_one_scenario(
            arch, P_s, Q_s, pv_s, wt_s,
            self.parent, self.children, self.z, self.post,
            export_tol_kwh=export_tol_kwh,
            fc_pmin_frac=fc_pmin_frac
        )

    
    def evaluate_architecture(
        self,
        x: np.ndarray,
        base_arch: Dict[str, float],
        scenario_cfg: ScenarioConfig,
        cost_cfg: CostConfig,
        arch_id: str = "ARCH",
        n_workers: int = 0,
        export_tol_kwh: float = 0.0,
        fc_pmin_frac: float = FC_PMIN_FRAC_DEFAULT
    ) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
        """
        Returns:
          obj: dict with raw means
          met: dict with mean/std/worst + cost breakdown
          df_per: per-scenario dataframe
        """
        arch = arch_from_vector(x, base_arch)

        T = int(self.P_base.shape[0])
        assert T == 360, "Expected 360h horizon."

        
        if n_workers is None or n_workers <= 1:
            per = [self.run_one_scenario(arch, s, scenario_cfg, export_tol_kwh=export_tol_kwh, fc_pmin_frac=fc_pmin_frac)
                   for s in range(scenario_cfg.N)]
        else:
          
            per = _mp_run(self, arch, scenario_cfg, n_workers, export_tol_kwh, fc_pmin_frac)

        df = pd.DataFrame(per)

        
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

        met: Dict[str, float] = {"Arch_ID": arch_id}

        # Core stats
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
        ]:
            m, s, w = _mean_std_worst(col, worst_is=worst_is)
            met[f"{col}_mean"] = m
            met[f"{col}_std"]  = s
            met[f"{col}_worst"] = w

        # cost breakdown for mean point
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

def _mp_run(sim: Simulator, arch: Dict[str, float], cfg: ScenarioConfig, n_workers: int,
            export_tol_kwh: float, fc_pmin_frac: float):
    ctx = mp.get_context("spawn")
  
    per = [sim.run_one_scenario(arch, s, cfg, export_tol_kwh=export_tol_kwh, fc_pmin_frac=fc_pmin_frac)
           for s in range(cfg.N)]
    return per
