import numpy as np
import pandas as pd
from collections import defaultdict, deque

LINE_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\newlineparameters.xlsx"
LINE_SHEET = 0

LOAD_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\CIGRE_15day_Loads_R1toR18_CLEAN_S95cap.xlsx"
P_SHEET    = "Active_kW_R1toR18"
Q_SHEET    = "Reactive_kVAr_R1toR18"

OUT_XLSX   = r"C:\Users\Dell\Desktop\GRT\Codes\BFS_TimeSeries_NoDER_PU.xlsx"

SLACK_BUS  = "R1"


SBASE_KVA   = 400.0   # kVA
VLL_BASE_KV = 0.4     # kV


MAX_ITERS = 120
TOL_V_PU  = 1e-9


V_MIN_PU = 0.95
V_MAX_PU = 1.05


def norm_bus(x) -> str:
    """Normalize bus labels to 'R#' format (handles 1, 1.0, 'R1', ' r01 ', etc.)"""
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if s.startswith("R"):
        s2 = s[1:].strip()
        try:
            return f"R{int(float(s2))}"
        except:
            return "R" + s2
    try:
        return f"R{int(float(s))}"
    except:
        return s


def find_col_case_insensitive(df, candidates):
    
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def read_line_data(path, sheet=0):
 
  
    df_try = pd.read_excel(path, sheet_name=sheet)
    df_try.columns = [str(c).strip() for c in df_try.columns]

    fb = find_col_case_insensitive(df_try, ["from_bus", "from", "fbus"])
    tb = find_col_case_insensitive(df_try, ["to_bus", "to", "tbus"])
    rc = find_col_case_insensitive(df_try, ["r", "r_ohm", "resistance"])
    xc = find_col_case_insensitive(df_try, ["x", "x_ohm", "reactance"])

    if fb and tb and rc and xc:
        df = df_try[[fb, tb, rc, xc]].copy()
        df.columns = ["from_bus", "to_bus", "R", "X"]
    else:
        
        df = pd.read_excel(path, sheet_name=sheet, header=None).dropna(how="all")
        if df.shape[1] < 4:
            raise ValueError(f"Line file must have at least 4 columns. Found {df.shape[1]}")
        df = df.iloc[:, :4].copy()
        df.columns = ["from_bus", "to_bus", "R", "X"]

    df["from_bus"] = df["from_bus"].apply(norm_bus)
    df["to_bus"]   = df["to_bus"].apply(norm_bus)
    df = df[(df["from_bus"] != "") & (df["to_bus"] != "")]
    df = df.dropna(subset=["R", "X"])
    df["R"] = df["R"].astype(float)
    df["X"] = df["X"].astype(float)
    return df.reset_index(drop=True)


def ensure_day_hour(dfP, dfQ):
    
    dfP = dfP.copy()
    dfQ = dfQ.copy()

    dfP.columns = [str(c).strip() for c in dfP.columns]
    dfQ.columns = [str(c).strip() for c in dfQ.columns]

    dayP  = find_col_case_insensitive(dfP, ["day", "days"])
    hourP = find_col_case_insensitive(dfP, ["hour", "hours", "hr"])
    dayQ  = find_col_case_insensitive(dfQ, ["day", "days"])
    hourQ = find_col_case_insensitive(dfQ, ["hour", "hours", "hr"])

    if dayP and hourP and dayQ and hourQ:
        dfP = dfP.rename(columns={dayP: "Day", hourP: "Hour"})
        dfQ = dfQ.rename(columns={dayQ: "Day", hourQ: "Hour"})
        dfP["Day"] = dfP["Day"].astype(int)
        dfP["Hour"] = dfP["Hour"].astype(int)
        dfQ["Day"] = dfQ["Day"].astype(int)
        dfQ["Hour"] = dfQ["Hour"].astype(int)
        return dfP, dfQ

    
    tP = find_col_case_insensitive(dfP, ["t"])
    tQ = find_col_case_insensitive(dfQ, ["t"])
    if not (tP and tQ):
        raise ValueError(
            "Load sheets must contain Day/Hour (any case) OR a 't' column to derive them.\n"
            f"Active columns: {dfP.columns.tolist()}\n"
            f"Reactive columns: {dfQ.columns.tolist()}"
        )

    dfP = dfP.rename(columns={tP: "t"})
    dfQ = dfQ.rename(columns={tQ: "t"})

    def t_to_idx(x):
        s = str(x).strip().lower()
        if s.startswith("t"):
            return int(float(s[1:]))
        return int(float(s))

    dfP["_idx"] = dfP["t"].apply(t_to_idx)
    dfQ["_idx"] = dfQ["t"].apply(t_to_idx)

    dfP["Day"] = (dfP["_idx"] // 24) + 1
    dfP["Hour"] = dfP["_idx"] % 24
    dfQ["Day"] = (dfQ["_idx"] // 24) + 1
    dfQ["Hour"] = dfQ["_idx"] % 24

    dfP = dfP.drop(columns=["_idx"])
    dfQ = dfQ.drop(columns=["_idx"])

    dfP["Day"] = dfP["Day"].astype(int)
    dfP["Hour"] = dfP["Hour"].astype(int)
    dfQ["Day"] = dfQ["Day"].astype(int)
    dfQ["Hour"] = dfQ["Hour"].astype(int)

    return dfP, dfQ


def build_tree(line_df_pu: pd.DataFrame, slack: str):
    adj = defaultdict(list)
    z_undir = {}
    buses = set()

    for _, r in line_df_pu.iterrows():
        f, t = r["from_bus"], r["to_bus"]
        Rpu, Xpu = float(r["R_pu"]), float(r["X_pu"])
        buses.add(f); buses.add(t)
        adj[f].append(t); adj[t].append(f)
        z_undir[(f, t)] = (Rpu, Xpu)
        z_undir[(t, f)] = (Rpu, Xpu)

    if slack not in buses:
        raise ValueError(f"Slack bus {slack} not found. Found buses: {sorted(buses)}")

    parent = {slack: None}
    children = defaultdict(list)
    q = deque([slack])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                children[u].append(v)
                q.append(v)

    if len(parent) != len(buses):
        missing = sorted(buses - set(parent.keys()))
        raise ValueError(f"Network not fully connected from {slack}. Missing buses: {missing}")

    branch_map = {}
    for child, par in parent.items():
        if par is None:
            continue
        branch_map[(par, child)] = z_undir[(par, child)]

    
    order_fwd = []
    q = deque([slack])
    while q:
        u = q.popleft()
        for v in children[u]:
            order_fwd.append(v)
            q.append(v)

    
    order_bwd = []
    def post(u):
        for v in children[u]:
            post(v)
        if u != slack:
            order_bwd.append(u)
    post(slack)

    def bus_key(b):
        return int(b[1:]) if b.startswith("R") and b[1:].isdigit() else b
    buses_sorted = sorted(buses, key=bus_key)

    return buses_sorted, parent, children, branch_map, order_bwd, order_fwd


def bfs_one_step_pu(buses, parent, children, branch_map, order_bwd, order_fwd,
                    slack, Sbus_pu, Vslack_pu=1+0j):
    V = {b: 1+0j for b in buses}
    V[slack] = Vslack_pu
    non_slack = [b for b in buses if b != slack]

    for _ in range(MAX_ITERS):
        V_prev = V.copy()

        
        Iinj = {b: 0j for b in buses}
        for b in non_slack:
            S = Sbus_pu.get(b, 0j)
            if abs(V[b]) > 1e-12:
                Iinj[b] = np.conj(S) / np.conj(V[b])
            else:
                Iinj[b] = 0j

       
        Itotal = {b: Iinj.get(b, 0j) for b in buses}
        Ibr = {b: 0j for b in non_slack} 

        for b in order_bwd:
            par = parent[b]
            Ibr[b] = Itotal[b]
            Itotal[par] += Itotal[b]

        
        V[slack] = Vslack_pu
        for b in order_fwd:
            par = parent[b]
            Rpu, Xpu = branch_map[(par, b)]
            Zpu = Rpu + 1j * Xpu
            V[b] = V[par] - Zpu * Ibr[b]

        dv = max(abs(V[b] - V_prev[b]) for b in buses)
        if dv < TOL_V_PU:
            break

    
    Ploss_pu = 0.0
    Qloss_pu = 0.0
    for b in non_slack:
        par = parent[b]
        Rpu, Xpu = branch_map[(par, b)]
        I = Ibr[b]
        Ploss_pu += (abs(I)**2) * Rpu
        Qloss_pu += (abs(I)**2) * Xpu

    return V, Ibr, Ploss_pu, Qloss_pu



def run_time_series_bfs_pu():
   
    Sbase_MVA = SBASE_KVA / 1000.0
    Zbase_ohm = (VLL_BASE_KV**2) / Sbase_MVA  # ohm
    print(f"Zbase = {Zbase_ohm:.6f} ohm  (VLL={VLL_BASE_KV} kV, S={SBASE_KVA} kVA)")

  
    line_df = read_line_data(LINE_XLSX, sheet=LINE_SHEET)
    line_pu = line_df.copy()
    line_pu["R_pu"] = line_pu["R"] / Zbase_ohm
    line_pu["X_pu"] = line_pu["X"] / Zbase_ohm

    buses, parent, children, branch_map, order_bwd, order_fwd = build_tree(line_pu, SLACK_BUS)

    
    dfP = pd.read_excel(LOAD_XLSX, sheet_name=P_SHEET)
    dfQ = pd.read_excel(LOAD_XLSX, sheet_name=Q_SHEET)

    
    dfP = dfP.rename(columns={c: norm_bus(c) for c in dfP.columns})
    dfQ = dfQ.rename(columns={c: norm_bus(c) for c in dfQ.columns})

   
    dfP, dfQ = ensure_day_hour(dfP, dfQ)

    dfP = dfP.sort_values(["Day", "Hour"]).reset_index(drop=True)
    dfQ = dfQ.sort_values(["Day", "Hour"]).reset_index(drop=True)

    nsteps = min(len(dfP), len(dfQ))
    if nsteps == 0:
        raise ValueError("No time steps found in load sheets.")

    
    bus_cols = [b for b in buses if b in dfP.columns and b in dfQ.columns]

   
    V_records = []
    loss_records = []
    summary_records = []
    vio_count_per_bus = {b: 0 for b in buses}
    total_hours = 0

    for k in range(nsteps):
        day = int(dfP.loc[k, "Day"])
        hour = int(dfP.loc[k, "Hour"])

        
        Sbus_pu = {}
        for b in bus_cols:
            PkW = float(dfP.loc[k, b])
            QkVAr = float(dfQ.loc[k, b])
            Sbus_pu[b] = (PkW + 1j * QkVAr) / SBASE_KVA

        Vpu, Ibr_pu, Ploss_pu, Qloss_pu = bfs_one_step_pu(
            buses=buses,
            parent=parent,
            children=children,
            branch_map=branch_map,
            order_bwd=order_bwd,
            order_fwd=order_fwd,
            slack=SLACK_BUS,
            Sbus_pu=Sbus_pu,
            Vslack_pu=1+0j
        )

        
        rowV = {"Day": day, "Hour": hour}
        Vmag_list = []
        for b in buses:
            vm = abs(Vpu[b])
            rowV[b] = vm
            Vmag_list.append(vm)
        V_records.append(rowV)

        
        loss_records.append({
            "Day": day, "Hour": hour,
            "P_loss_pu": Ploss_pu,
            "Q_loss_pu": Qloss_pu,
            "P_loss_kW": Ploss_pu * SBASE_KVA,
            "Q_loss_kVAr": Qloss_pu * SBASE_KVA,
            "P_loss_kWh_this_hour": Ploss_pu * SBASE_KVA
        })

        
        Vmag = np.array(Vmag_list)
        vio_mask = (Vmag < V_MIN_PU) | (Vmag > V_MAX_PU)
        vio_count = int(np.sum(vio_mask))
        for i, b in enumerate(buses):
            if vio_mask[i]:
                vio_count_per_bus[b] += 1

        total_hours += 1
        summary_records.append({
            "Day": day, "Hour": hour,
            "V_min_pu": float(Vmag.min()),
            "V_max_pu": float(Vmag.max()),
            "Voltage_violation_bus_count": vio_count,
            "Voltage_violation_exists": int(vio_count > 0),
            "P_loss_kW": Ploss_pu * SBASE_KVA
        })

    V_df = pd.DataFrame(V_records)
    loss_df = pd.DataFrame(loss_records)
    summary_df = pd.DataFrame(summary_records)

    
    vio_stats = []
    for b in buses:
        pct = (vio_count_per_bus[b] / total_hours * 100.0) if total_hours else 0.0
        vio_stats.append([b, vio_count_per_bus[b], pct])
    vio_stats_df = pd.DataFrame(vio_stats, columns=["Bus", "ViolationHoursCount", "ViolationHoursPercent"])

    
    overall_df = pd.DataFrame([{
        "TotalHours": total_hours,
        "TotalEnergyLoss_kWh": float(loss_df["P_loss_kWh_this_hour"].sum()),
        "AvgLoss_kW": float(loss_df["P_loss_kW"].mean()),
        "MaxLoss_kW": float(loss_df["P_loss_kW"].max()),
        "HoursWithAnyViolation": int(summary_df["Voltage_violation_exists"].sum()),
        "PercentHoursWithAnyViolation": float(summary_df["Voltage_violation_exists"].mean() * 100.0)
    }])

    
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        line_pu.to_excel(writer, sheet_name="LineData_pu", index=False)
        V_df.to_excel(writer, sheet_name="BusVoltages_pu", index=False)
        loss_df.to_excel(writer, sheet_name="Losses", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        vio_stats_df.to_excel(writer, sheet_name="ViolationStats_perBus", index=False)
        overall_df.to_excel(writer, sheet_name="OverallStats", index=False)

    print("✅ Per-unit BFS finished successfully.")
    print("Saved:", OUT_XLSX)


if __name__ == "__main__":
    run_time_series_bfs_pu()
