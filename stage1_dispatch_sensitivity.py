import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import time


PATH_LINES_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\newlineparameters.xlsx"

PATH_LOAD_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\CIGRE_15day_Loads_R1toR18_CLEAN_S95cap.xlsx"
SHEET_PLOAD     = "Active_kW_R1toR18"
SHEET_QLOAD     = "Reactive_kVAr_R1toR18"

PATH_PV_XLSX    = r"C:\Users\Dell\Desktop\GRT\Codes\PV_15day_Hourly_Profile_Khumaltar_NOCT.xlsx"
SHEET_PV        = "PV_Hourly_15days"
PV_PER_KW_COL   = "PV_kW_per_1kW_STC"

PATH_ALT_XLSX   = r"C:\Users\Dell\Desktop\GRT\Codes\Alternatives_CIGRE_R1toR18.xlsx"
SHEET_ALT       = "Alternatives"

OUT_DIR         = r"C:\Users\Dell\Desktop\GRT\Codes\Results of stage 1\Sensitivity"
CASE_TAG        = "Stage1_DispatchSensitivity_BESSOnly"

VLL_kV    = 0.4
Sbase_kVA = 400.0
SLACK_BUS = "R1"
Vslack_pu = 1.0 + 0j

Vmin_lim = 0.95
Vmax_lim = 1.05

ALL_BUSES = [f"R{i}" for i in range(1, 19)]


ENFORCE_NO_EXPORT = True

=
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

@dataclass
class ScenarioConfig:
    N: int = 50
    seed: int = 7

    load_daily_sigma: float = 0.06
    load_hourly_sigma: float = 0.03
    load_clip: tuple = (0.85, 1.20)

    pv_daily_sigma: float = 0.12
    pv_hourly_sigma: float = 0.06
    pv_clip: tuple = (0.0, 1.25)

    wt_daily_sigma: float = 0.15
    wt_hourly_sigma: float = 0.10
    wt_clip: tuple = (0.0, 1.40)

    enable_pv_dips: bool = True
    pv_dip_prob_per_day: float = 0.15
    pv_dip_hours: tuple = (11, 15)
    pv_dip_strength: tuple = (0.60, 0.85)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_bus_name(x):
    s = str(x).strip()
    if not s:
        return s
    s = s.replace(" ", "")
    if s.upper().startswith("R"):
        return "R" + s[1:].strip()
    try:
        n = int(float(s))
        return f"R{n}"
    except:
        return s

def bi(busname: str) -> int:
    return int(busname[1:]) - 1

def safe_to_excel(df, path, index=False):
    try:
        df.to_excel(path, index=index)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        path2 = base + "_try2" + ext
        df.to_excel(path2, index=index)
        return path2


def read_lines(path):
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

def build_tree_strict(lines, slack="R1"):
    buses = sorted(set(lines["from_bus"]).union(set(lines["to_bus"])).union(set(ALL_BUSES)))
    if slack not in buses:
        raise ValueError(f"Slack '{slack}' not present in line data buses: {buses}")

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
        raise ValueError(f"Network is NOT fully connected from {slack}. Unreachable buses: {missing}")

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

def read_bus_matrix_from_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    cols = [c for c in df.columns if str(c).strip().upper().startswith("R")]
    if not cols:
        raise ValueError(f"Sheet '{sheet}' has no R* columns. Found: {df.columns.tolist()}")

    cols_norm = [normalize_bus_name(c) for c in cols]
    df = df[cols].copy()
    df.columns = cols_norm

    if len(df) < 360:
        raise ValueError(f"Sheet '{sheet}' has {len(df)} rows; need >=360.")

    df = df.iloc[:360].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for b in ALL_BUSES:
        if b not in df.columns:
            df[b] = 0.0
    df = df[ALL_BUSES]
    return df.values.astype(float)

def read_pv_per_kw_profile(path, sheet, col):
    df = pd.read_excel(path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"PV sheet '{sheet}' missing '{col}'. Found: {df.columns.tolist()}")
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
    cf = wind_power_cf(v)
    return cf.astype(float), v.astype(float)

def read_alternatives_exact(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    required = [
        "AltID",
        "PV_R16_kW", "PV_R17_kW", "WT_R16_kW",
        "BESS_R4_P_kW", "BESS_R4_E_kWh",
        "MT_R15_Pmax_kW", "FC_R18_kW"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Alternatives sheet missing columns: {missing}\nFound: {df.columns.tolist()}")

    out = []
    for _, r in df.iterrows():
        out.append({
            "Alt_ID": str(r["AltID"]).strip(),
            "PV16_kW": float(r["PV_R16_kW"]),
            "PV17_kW": float(r["PV_R17_kW"]),
            "WT16_kW": float(r["WT_R16_kW"]),
            "BESS_P_kW": float(r["BESS_R4_P_kW"]),
            "BESS_E_kWh": float(r["BESS_R4_E_kWh"]),
            "MT_kW": float(r["MT_R15_Pmax_kW"]),
            "FC_kW": float(r["FC_R18_kW"]),
        })
    return out


def bfs_solve(Sbus_pu, buses, parent, children, z, post, slack="R1",
              max_iter=80, tol=1e-8):
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

    return V, Ibr, Ploss_pu

def dispatch_bess_basic(hour_of_day, Pnet_prebess_kW, soc, Pmax_kW, E_kWh):
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


def dispatch_bess_smart(hour_of_day, Pnet_prebess_kW, soc, Pmax_kW, E_kWh):

    if Pmax_kW <= 1e-9 or E_kWh <= 1e-9:
        return 0.0, soc

    if hour_of_day in PEAK_HOURS and Pnet_prebess_kW > 0 and soc > SOC_MIN + 1e-6:
        E_avail = (soc - SOC_MIN) * E_kWh
        P_energy_limit = E_avail * ETA_DIS
        Pdis = min(Pmax_kW, Pnet_prebess_kW, P_energy_limit)
        soc_next = soc - (Pdis / ETA_DIS) / E_kWh
        return Pdis, max(SOC_MIN, soc_next)

    
    if Pnet_prebess_kW >= 0.25 * max(Pmax_kW, 1e-9) and soc > SOC_MIN + 1e-6:
        E_avail = (soc - SOC_MIN) * E_kWh
        P_energy_limit = E_avail * ETA_DIS
        Pdis = min(0.50 * Pmax_kW, Pnet_prebess_kW, P_energy_limit)
        if Pdis > 1e-9:
            soc_next = soc - (Pdis / ETA_DIS) / E_kWh
            return Pdis, max(SOC_MIN, soc_next)

    
    if Pnet_prebess_kW < 0 and soc < SOC_MAX - 1e-6:
        E_room = (SOC_MAX - soc) * E_kWh
        P_energy_limit = E_room / ETA_CH
        Pch = min(Pmax_kW, abs(Pnet_prebess_kW), P_energy_limit)
        if Pch > 1e-9:
            soc_next = soc + (ETA_CH * Pch) / E_kWh
            return -Pch, min(SOC_MAX, soc_next)

    
    if 10 <= hour_of_day <= 15 and Pnet_prebess_kW <= 0.20 * max(Pmax_kW, 1e-9) and soc < SOC_MAX - 1e-6:
        E_room = (SOC_MAX - soc) * E_kWh
        P_energy_limit = E_room / ETA_CH
        Pch = min(0.50 * Pmax_kW, P_energy_limit)
        if Pch > 1e-9:
            soc_next = soc + (ETA_CH * Pch) / E_kWh
            return -Pch, min(SOC_MAX, soc_next)

    return 0.0, soc

def extra_charge_bess_for_noexport(Pexport_kW, soc, Pmax_kW, E_kWh):
    if Pexport_kW <= 1e-9 or Pmax_kW <= 1e-9 or E_kWh <= 1e-9:
        return 0.0, soc
    if soc >= SOC_MAX - 1e-6:
        return 0.0, soc

    E_room = (SOC_MAX - soc) * E_kWh
    P_energy_limit = E_room / ETA_CH
    Pch = min(Pmax_kW, Pexport_kW, P_energy_limit)
    soc_next = soc + (ETA_CH * Pch) / E_kWh
    return Pch, min(SOC_MAX, soc_next)

def dispatch_microturbine_load_follow(P_remaining_kW, Pmax_kW, Pmin_frac=0.30):
    if Pmax_kW <= 1e-9 or P_remaining_kW <= 0:
        return 0.0
    Pmin = Pmin_frac * Pmax_kW
    if P_remaining_kW < Pmin:
        return 0.0
    return min(P_remaining_kW, Pmax_kW)

def curtail_renewables(Pneeded_kW, Ppv16, Ppv17, Pwt16):
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


def evaluate_one_alternative_with_dispatch(
    alt,
    Pmat_kW, Qmat_kVAr,
    pv_per1kw, wt_per1kw,
    buses, parent, children, z, post,
    dispatch_mode="BASELINE",
):
    PV16 = alt["PV16_kW"]
    PV17 = alt["PV17_kW"]
    WT16 = alt["WT16_kW"]
    PB   = alt["BESS_P_kW"]
    EB   = alt["BESS_E_kWh"]
    PMT  = alt["MT_kW"]
    PFC  = alt["FC_kW"]

    soc = SOC0

    total_loss_kWh = 0.0
    violation_bus_hours = 0
    minV_global, maxV_global = 999.0, 0.0

    E_grid_import = 0.0
    E_grid_export = 0.0
    E_pv_used = 0.0
    E_wt_used = 0.0
    E_pv_curt = 0.0
    E_wt_curt = 0.0
    E_bess_through = 0.0

    for t in range(360):
        hod = t % 24

        Pload_bus = Pmat_kW[t, :]
        Qload_bus = Qmat_kVAr[t, :]

        Ppv16_raw = PV16 * pv_per1kw[t]
        Ppv17_raw = PV17 * pv_per1kw[t]
        Pwt16_raw = WT16 * wt_per1kw[t]
        Pfc18 = PFC

        Ppv16 = Ppv16_raw
        Ppv17 = Ppv17_raw
        Pwt16 = Pwt16_raw

        Pgen = np.zeros(18, dtype=float)
        Qgen = np.zeros(18, dtype=float)

        Pgen[bi("R16")] += (Ppv16 + Pwt16)
        Pgen[bi("R17")] += Ppv17
        Pgen[bi("R18")] += Pfc18

        Pload_total = float(np.sum(Pload_bus))
        Pgen_total_prebess = float(np.sum(Pgen))
        Pnet_prebess = Pload_total - Pgen_total_prebess

        if dispatch_mode.upper() == "BASELINE":
            Pbess4, soc_next = dispatch_bess_basic(hod, Pnet_prebess, soc, PB, EB)
        else:
            Pbess4, soc_next = dispatch_bess_smart(hod, Pnet_prebess, soc, PB, EB)

        soc = soc_next
        Pgen[bi("R4")] += Pbess4

        Pgen_total_preMT = float(np.sum(Pgen))
        Pnet_preMT = Pload_total - Pgen_total_preMT
        Pmt15 = dispatch_microturbine_load_follow(Pnet_preMT, PMT, Pmin_frac=0.30)
        Pgen[bi("R15")] += Pmt15

        Pgen_total = float(np.sum(Pgen))
        Pnet_total = Pload_total - Pgen_total
        Pgrid_import = max(Pnet_total, 0.0)
        Pgrid_export = max(-Pnet_total, 0.0)

        c_pv16 = c_pv17 = c_wt16 = 0.0

        if ENFORCE_NO_EXPORT and Pgrid_export > 1e-9:
            extra_ch, soc2 = extra_charge_bess_for_noexport(Pgrid_export, soc, PB, EB)
            if extra_ch > 0:
                Pgen[bi("R4")] += (-extra_ch)
                soc = soc2

            Pgen_total = float(np.sum(Pgen))
            Pnet_total = Pload_total - Pgen_total
            Pgrid_export = max(-Pnet_total, 0.0)

            if Pgrid_export > 1e-9:
                c_pv16, c_pv17, c_wt16, Ppv16_new, Ppv17_new, Pwt16_new = curtail_renewables(
                    Pgrid_export, Ppv16, Ppv17, Pwt16
                )

                Pgen[bi("R16")] -= (Ppv16 + Pwt16)
                Pgen[bi("R17")] -= Ppv17

                Ppv16, Ppv17, Pwt16 = Ppv16_new, Ppv17_new, Pwt16_new

                Pgen[bi("R16")] += (Ppv16 + Pwt16)
                Pgen[bi("R17")] += Ppv17

                Pgen_total = float(np.sum(Pgen))
                Pnet_total = Pload_total - Pgen_total
                Pgrid_export = max(-Pnet_total, 0.0)

        Pgrid_import = max(Pnet_total, 0.0)

        Pnet_bus = Pload_bus - Pgen
        Qnet_bus = Qload_bus - Qgen

        Sbus_pu = {}
        for i in range(18):
            bus = f"R{i+1}"
            if bus == SLACK_BUS:
                continue
            S_kVA = (Pnet_bus[i] + 1j * Qnet_bus[i])
            Sbus_pu[bus] = S_kVA / Sbase_kVA

        V, Ibr, Ploss_pu = bfs_solve(Sbus_pu, ALL_BUSES, parent, children, z, post, slack=SLACK_BUS)
        Ploss_kW = Ploss_pu * Sbase_kVA
        total_loss_kWh += Ploss_kW

        Vmag = np.array([abs(V[b]) for b in ALL_BUSES], dtype=float)
        Vmin = float(Vmag.min())
        Vmax = float(Vmag.max())
        minV_global = min(minV_global, Vmin)
        maxV_global = max(maxV_global, Vmax)
        violation_bus_hours += int(np.sum((Vmag < Vmin_lim) | (Vmag > Vmax_lim)))

        E_grid_import += Pgrid_import
        E_grid_export += max(Pgrid_export, 0.0)

        E_pv_used += (Ppv16 + Ppv17)
        E_wt_used += Pwt16
        E_pv_curt += (c_pv16 + c_pv17)
        E_wt_curt += c_wt16
        E_bess_through += abs(Pbess4)

    bess_cycles = (E_bess_through / (2.0 * EB)) if EB > 1e-9 else 0.0
    E_ren_avail = (E_pv_used + E_pv_curt) + (E_wt_used + E_wt_curt)
    E_ren_used  = E_pv_used + E_wt_used
    ren_util_pct = 100.0 * (E_ren_used / E_ren_avail) if E_ren_avail > 1e-9 else 0.0
    curt_kWh = E_pv_curt + E_wt_curt

    metrics = {
        "Loss_kWh": float(total_loss_kWh),
        "Vmin_pu": float(minV_global),
        "Vmax_pu": float(maxV_global),
        "ViolBH": float(violation_bus_hours),
        "Import_kWh": float(E_grid_import),
        "Export_kWh": float(E_grid_export),
        "RenUtil_pct": float(ren_util_pct),
        "Curt_kWh": float(curt_kWh),
        "BESS_cycles": float(bess_cycles),
    }
    return metrics


def _lognormal_mult(rng, sigma, size):
    x = rng.normal(0.0, sigma, size=size)
    return np.exp(x - 0.5 * sigma * sigma)

def make_scenario_multipliers(T, cfg: ScenarioConfig):
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

    return {"load_mult": load_mult, "pv_mult": pv_mult, "wt_mult": wt_mult}

def aggregate_metrics_over_scenarios(metrics_list, metric_specs):
    keys = list(metric_specs.keys())
    arr = {k: np.array([m[k] for m in metrics_list], dtype=float) for k in keys}

    out = {}
    for k in keys:
        out[f"{k}_mean"] = float(np.mean(arr[k]))
        if metric_specs[k] == "cost":
            out[f"{k}_worst"] = float(np.max(arr[k]))
        else:
            out[f"{k}_worst"] = float(np.min(arr[k]))
    return out

def run_optionB_dispatch_sensitivity(
    alts,
    P_base, Q_base,
    pv_base_per1kw,
    wt_base_per1kw,
    buses, parent, children, z, post,
    cfg: ScenarioConfig,
    dispatch_mode: str,
):
    metric_specs = {
        "Loss_kWh": "cost",
        "Vmin_pu": "benefit",
        "ViolBH": "cost",
        "Import_kWh": "cost",
        "RenUtil_pct": "benefit",
        "Curt_kWh": "cost",
        "BESS_cycles": "cost",
    }

    T = P_base.shape[0]
    mult = make_scenario_multipliers(T, cfg)
    load_mult = mult["load_mult"]
    pv_mult   = mult["pv_mult"]
    wt_mult   = mult["wt_mult"]

    robust_rows = []
    scenario_rows = []

    for alt in alts:
        alt_id = alt["Alt_ID"]
        print(f"\n=== {dispatch_mode}: {alt_id} ({cfg.N} scenarios) ===")

        per_s_metrics = []
        for s in range(cfg.N):
            P_s = P_base * load_mult[s, :, None]
            Q_s = Q_base * load_mult[s, :, None]
            pv_s = pv_base_per1kw * pv_mult[s, :]
            wt_s = wt_base_per1kw * wt_mult[s, :]

            m = evaluate_one_alternative_with_dispatch(
                alt=alt,
                Pmat_kW=P_s,
                Qmat_kVAr=Q_s,
                pv_per1kw=pv_s,
                wt_per1kw=wt_s,
                buses=buses,
                parent=parent,
                children=children,
                z=z,
                post=post,
                dispatch_mode=dispatch_mode,
            )

            per_s_metrics.append(m)
            scenario_rows.append({"DispatchMode": dispatch_mode, "Alt_ID": alt_id, "Scenario": s, **m})

        robust = aggregate_metrics_over_scenarios(per_s_metrics, metric_specs)
        robust_rows.append({"DispatchMode": dispatch_mode, "Alt_ID": alt_id, **robust})

    robust_df = pd.DataFrame(robust_rows)
    scen_df   = pd.DataFrame(scenario_rows)
    return robust_df, scen_df, load_mult, pv_mult, wt_mult


def make_quick_rank_compare(df_base, df_smart):
   
    metrics = [
        ("Loss_kWh_mean", "cost"),
        ("Vmin_pu_mean", "benefit"),
        ("ViolBH_mean", "cost"),
        ("Import_kWh_mean", "cost"),
        ("RenUtil_pct_mean", "benefit"),
        ("Curt_kWh_mean", "cost"),
        ("BESS_cycles_mean", "cost"),
    ]

    def score(df):
        X = df.copy().set_index("Alt_ID")
        s = pd.Series(0.0, index=X.index)
        for col, typ in metrics:
            x = X[col].astype(float)
            xmin, xmax = float(x.min()), float(x.max())
            if abs(xmax - xmin) < 1e-12:
                xn = pd.Series(1.0, index=x.index)
            else:
                if typ == "benefit":
                    xn = (x - xmin) / (xmax - xmin)
                else:
                    xn = (xmax - x) / (xmax - xmin)
            s += xn
        s = s / len(metrics)
        out = s.sort_values(ascending=False).reset_index()
        out.columns = ["Alt_ID", "QuickScore"]
        out["Rank"] = np.arange(1, len(out) + 1)
        return out

    r0 = score(df_base)
    r1 = score(df_smart)

    comp = r0.merge(r1, on="Alt_ID", how="outer", suffixes=("_Baseline", "_Smart"))
    comp["RankShift"] = comp["Rank_Smart"] - comp["Rank_Baseline"]

    shortlist = {"A1", "A2", "A14"}
    comp["InShortlist_BaselineLogic"] = comp["Alt_ID"].isin(shortlist).astype(int)
    return comp.sort_values(["Rank_Baseline", "Alt_ID"]).reset_index(drop=True)


def main():
    ensure_dir(OUT_DIR)
    t0_main = time.time()

    lines = read_lines(PATH_LINES_XLSX)
    VLL = VLL_kV * 1e3
    Sbase = Sbase_kVA * 1e3
    Zbase = (VLL**2) / Sbase
    lines["Zpu"] = (lines["R"].values + 1j*lines["X"].values) / Zbase

    buses, parent, children, z, post = build_tree_strict(lines, slack=SLACK_BUS)
    print(f"Topology OK: all buses reachable from {SLACK_BUS}")

    Pmat_kW = read_bus_matrix_from_sheet(PATH_LOAD_XLSX, SHEET_PLOAD)
    Qmat_kVAr = read_bus_matrix_from_sheet(PATH_LOAD_XLSX, SHEET_QLOAD)
    print(f"Loaded load sheets: P='{SHEET_PLOAD}', Q='{SHEET_QLOAD}' (360 hours)")

    pv_per_kw = read_pv_per_kw_profile(PATH_PV_XLSX, SHEET_PV, PV_PER_KW_COL)
    print(f"Loaded PV: sheet='{SHEET_PV}', col='{PV_PER_KW_COL}' (360 hours)")

    wind_cf, wind_v = build_wind_cf_360()
    wind_df = pd.DataFrame({
        "t": np.arange(360),
        "Day": (np.arange(360)//24)+1,
        "Hour": np.arange(360)%24,
        "wind_speed_mps": wind_v,
        "wind_capacity_factor": wind_cf
    })
    wind_path = safe_to_excel(wind_df, os.path.join(OUT_DIR, "Wind_15day_hourly_synthetic_dispatch_sensitivity.xlsx"))
    print("Saved wind profile:", wind_path)

    alts_all = read_alternatives_exact(PATH_ALT_XLSX, SHEET_ALT)
    print(f"Loaded {len(alts_all)} alternatives from '{SHEET_ALT}'")

    BESS_ALT_IDS = {"A3", "A4", "A5", "A6", "A7", "A8", "A10", "A11", "A12", "A13", "A14"}
    alts = [a for a in alts_all if a["Alt_ID"] in BESS_ALT_IDS]
    print(f"Dispatch sensitivity will run on {len(alts)} BESS architectures: {[a['Alt_ID'] for a in alts]}")

    cfg = ScenarioConfig(N=50, seed=7)

    robust_base, scen_base, load_mult, pv_mult, wt_mult = run_optionB_dispatch_sensitivity(
        alts=alts,
        P_base=Pmat_kW,
        Q_base=Qmat_kVAr,
        pv_base_per1kw=pv_per_kw,
        wt_base_per1kw=wind_cf,
        buses=buses, parent=parent, children=children, z=z, post=post,
        cfg=cfg,
        dispatch_mode="BASELINE",
    )

    robust_smart, scen_smart, _, _, _ = run_optionB_dispatch_sensitivity(
        alts=alts,
        P_base=Pmat_kW,
        Q_base=Qmat_kVAr,
        pv_base_per1kw=pv_per_kw,
        wt_base_per1kw=wind_cf,
        buses=buses, parent=parent, children=children, z=z, post=post,
        cfg=cfg,
        dispatch_mode="SMART_HEURISTIC",
    )

    quick_compare = make_quick_rank_compare(robust_base, robust_smart)

    t1_main = time.time()
    runtime_s = t1_main - t0_main

    runinfo_df = pd.DataFrame([{
        "CaseTag": CASE_TAG,
        "Runtime_s": runtime_s,
        "Runtime_min": runtime_s / 60.0,
        "Runtime_hr": runtime_s / 3600.0,
        "ScenarioCount": cfg.N,
        "ScenarioSeed": cfg.seed,
        "NumAlternatives_Rerun": len(alts),
        "DispatchModesCompared": "BASELINE vs SMART_HEURISTIC",
        "NumHours": int(Pmat_kW.shape[0]),
        "ENFORCE_NO_EXPORT": int(ENFORCE_NO_EXPORT),
        "Vmin_lim": Vmin_lim,
        "Vmax_lim": Vmax_lim,
        "CPU_or_System": "",
        "Notes": "Stage-1 dispatch sensitivity runtime"
    }])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{CASE_TAG}_OptionB_50scen_{stamp}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        robust_base.to_excel(w, sheet_name="DecisionMatrix_Baseline", index=False)
        robust_smart.to_excel(w, sheet_name="DecisionMatrix_Smart", index=False)
        scen_base.to_excel(w, sheet_name="ScenarioMetrics_Baseline", index=False)
        scen_smart.to_excel(w, sheet_name="ScenarioMetrics_Smart", index=False)
        quick_compare.to_excel(w, sheet_name="RankShift_QuickCompare", index=False)
        pd.DataFrame(load_mult).to_excel(w, sheet_name="LoadMultipliers", index=False)
        pd.DataFrame(pv_mult).to_excel(w, sheet_name="PVMultipliers", index=False)
        pd.DataFrame(wt_mult).to_excel(w, sheet_name="WTMultipliers", index=False)
        wind_df.to_excel(w, sheet_name="WindBaseProfile", index=False)
        runinfo_df.to_excel(w, sheet_name="RunInfo", index=False)

    print("\n✅ Dispatch sensitivity study DONE.")
    print("✅ Saved Excel workbook:", out_path)
    print(f"Stage-1 dispatch sensitivity runtime = {runtime_s:.1f} s = {(runtime_s/60):.2f} min")


if __name__ == "__main__":
    main()