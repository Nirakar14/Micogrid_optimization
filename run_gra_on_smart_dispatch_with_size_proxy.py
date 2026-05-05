import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

DISPATCH_SENS_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\Results of stage 1\Stage1_DispatchSensitivity_BESSOnly_OptionB_50scen_20260403_082409.xlsx"

ALT_XLSX = r"C:\Users\Dell\Desktop\GRT\Codes\Alternatives_CIGRE_R1toR18.xlsx"
ALT_SHEET = "Alternatives"


BASELINE_GRA_XLSX = r""

OUT_DIR = r"C:\Users\Dell\Desktop\GRT\Codes\Result_gra_after_size_cost_wrostremove"


SMART_DM_SHEET = "DecisionMatrix_Smart"
RANK_MODE = "MEAN_WORST"  
BASE_ZETA = 0.5
ZETA_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]
TOPK = 5

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
})

METRIC_TYPE = {
    "Loss_kWh": "cost",
    "Vmin_pu": "benefit",
    "ViolBH": "cost",
    "Import_kWh": "cost",
    "RenUtil_pct": "benefit",
    "Curt_kWh": "cost",
    "BESS_cycles": "cost",
    "InstalledCapacity_kW": "cost",
    "StorageEnergy_kWh": "cost",
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def base_metric_name(col):
    if col.endswith("_mean"):
        return col[:-5]
    if col.endswith("_worst"):
        return col[:-6]
    return col

def infer_metric_type(col):
    b = base_metric_name(col)
    return METRIC_TYPE.get(b, "cost")

def minmax_normalize(df, types):
    norm = df.copy()
    for j, c in enumerate(df.columns):
        x = df[c].astype(float).values
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if np.isclose(xmax, xmin):
            norm[c] = 1.0
            continue
        if types[j] == "benefit":
            norm[c] = (x - xmin) / (xmax - xmin)
        else:
            norm[c] = (xmax - x) / (xmax - xmin)
        norm[c] = np.clip(norm[c], 0.0, 1.0)
    return norm

def gra_compute(norm_df, weights=None, zeta=0.5):
    X = norm_df.values.astype(float)
    ref = np.ones((1, X.shape[1]), dtype=float)

    delta = np.abs(ref - X)
    dmin = np.min(delta)
    dmax = np.max(delta)

    coeff = (dmin + zeta * dmax) / (delta + zeta * dmax + 1e-12)

    if weights is None:
        weights = np.ones(X.shape[1], dtype=float) / X.shape[1]
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)

    grade = coeff @ weights
    coeff_df = pd.DataFrame(coeff, index=norm_df.index, columns=norm_df.columns)
    grade_s = pd.Series(grade, index=norm_df.index, name="GRA_Grade")
    return coeff_df, grade_s

def rank_from_grade(grade_s):
    df = grade_s.sort_values(ascending=False).reset_index()
    df.columns = ["Alt_ID", "GRA_Grade"]
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

def ieee_axes_style(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_plot(fig, outbase):
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)

def clean_alt_id(x):
    return str(x).strip()

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nFound: {df.columns.tolist()}")

def build_weight_sets(criteria_cols):
    n = len(criteria_cols)
    equal = np.ones(n) / n

    def make_boost(boost_bases, boost_factor=3.0):
        w = np.ones(n, dtype=float)
        for j, col in enumerate(criteria_cols):
            b = base_metric_name(col)
            if b in boost_bases:
                w[j] *= boost_factor
        return w / np.sum(w)

    sets = {
        "EqualWeights": equal,
        "VoltagePriority": make_boost({"Vmin_pu", "ViolBH"}, 3.0),
        "RenewablePriority": make_boost({"RenUtil_pct", "Curt_kWh"}, 3.0),
        "LossPriority": make_boost({"Loss_kWh"}, 4.0),
        "GridIndependence": make_boost({"Import_kWh"}, 4.0),
        "SizePenaltyStrong": make_boost({"InstalledCapacity_kW", "StorageEnergy_kWh"}, 5.0),
        "SizePenaltyMild": make_boost({"InstalledCapacity_kW", "StorageEnergy_kWh"}, 2.0),
    }
    return sets


def build_augmented_smart_dm():
    dm = pd.read_excel(DISPATCH_SENS_XLSX, sheet_name=SMART_DM_SHEET)
    if "Alt_ID" not in dm.columns:
        raise ValueError(f"'{SMART_DM_SHEET}' must contain Alt_ID. Found: {dm.columns.tolist()}")
    dm = dm.copy()
    dm["Alt_ID"] = dm["Alt_ID"].apply(clean_alt_id)

    alt = pd.read_excel(ALT_XLSX, sheet_name=ALT_SHEET)
    needed_alt_cols = [
        "AltID",
        "PV_R16_kW", "PV_R17_kW", "WT_R16_kW",
        "BESS_R4_P_kW", "BESS_R4_E_kWh",
        "MT_R15_Pmax_kW", "FC_R18_kW"
    ]
    require_cols(alt, needed_alt_cols, f"Alternatives sheet '{ALT_SHEET}'")
    alt = alt[needed_alt_cols].copy()
    alt["AltID"] = alt["AltID"].apply(clean_alt_id)

    alt["InstalledCapacity_kW"] = (
        alt["PV_R16_kW"].astype(float)
        + alt["PV_R17_kW"].astype(float)
        + alt["WT_R16_kW"].astype(float)
        + alt["MT_R15_Pmax_kW"].astype(float)
        + alt["FC_R18_kW"].astype(float)
        + alt["BESS_R4_P_kW"].astype(float)
    )
    alt["StorageEnergy_kWh"] = alt["BESS_R4_E_kWh"].astype(float)

    proxy = alt[["AltID", "InstalledCapacity_kW", "StorageEnergy_kWh"]].copy()
    proxy = proxy.rename(columns={"AltID": "Alt_ID"})

    out = dm.merge(proxy, on="Alt_ID", how="left")
    if out["InstalledCapacity_kW"].isna().any() or out["StorageEnergy_kWh"].isna().any():
        bad = out[out["InstalledCapacity_kW"].isna() | out["StorageEnergy_kWh"].isna()][["Alt_ID"]]
        raise ValueError(
            "Some Alt_ID did not match Alternatives.\n"
            f"Unmatched Alt_ID examples:\n{bad.head(10).to_string(index=False)}"
        )

    out["InstalledCapacity_kW_mean"]  = out["InstalledCapacity_kW"]
    out["InstalledCapacity_kW_worst"] = out["InstalledCapacity_kW"]
    out["StorageEnergy_kWh_mean"]     = out["StorageEnergy_kWh"]
    out["StorageEnergy_kWh_worst"]    = out["StorageEnergy_kWh"]

    return dm, proxy, out


def load_baseline_ranking():
    if not BASELINE_GRA_XLSX or not os.path.isfile(BASELINE_GRA_XLSX):
        return None
    xl = pd.ExcelFile(BASELINE_GRA_XLSX)
   
    for sheet in ["Ranking_Base", "Ranking", "BaseRanking"]:
        if sheet in xl.sheet_names:
            df = pd.read_excel(BASELINE_GRA_XLSX, sheet_name=sheet)
            if "Alt_ID" in df.columns and "Rank" in df.columns:
                return df[["Alt_ID", "GRA_Grade", "Rank"]].copy()
    return None


def main():
    ensure_dir(OUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_dm, proxy_df, aug_dm = build_augmented_smart_dm()


    aug_path = os.path.join(OUT_DIR, f"SmartDispatch_DM_PLUS_SizeProxy_{stamp}.xlsx")
    with pd.ExcelWriter(aug_path, engine="openpyxl") as w:
        aug_dm.to_excel(w, sheet_name="DecisionMatrix_MeanWorst", index=False)
        original_dm.to_excel(w, sheet_name="Original_Smart_DM", index=False)
        proxy_df.to_excel(w, sheet_name="SizeProxy_FromAlternatives", index=False)

    dm = aug_dm.copy()
    dm["Alt_ID"] = dm["Alt_ID"].astype(str).str.strip()
    dm = dm.set_index("Alt_ID")

    all_cols = [c for c in dm.columns if isinstance(c, str)]
    mean_cols = [c for c in all_cols if c.endswith("_mean")]
    worst_cols = [c for c in all_cols if c.endswith("_worst")]

    if RANK_MODE.upper() == "MEAN_ONLY":
        criteria_cols = mean_cols
    else:
        criteria_cols = mean_cols + worst_cols

    if not criteria_cols:
        raise ValueError("No criteria columns found ending with _mean/_worst.")

    Xraw = dm[criteria_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    crit_types = [infer_metric_type(c) for c in criteria_cols]
    Xnorm = minmax_normalize(Xraw, crit_types)

  
    coeff_df, grade_s = gra_compute(Xnorm, weights=None, zeta=BASE_ZETA)
    ranking_df = rank_from_grade(grade_s)

  
    zeta_rank_maps = {}
    for z in ZETA_SWEEP:
        _, g = gra_compute(Xnorm, weights=None, zeta=z)
        r = rank_from_grade(g)
        zeta_rank_maps[z] = r.set_index("Alt_ID")["Rank"].to_dict()


    weight_sets = build_weight_sets(criteria_cols)
    weight_rankings = {}
    for name, w in weight_sets.items():
        _, g = gra_compute(Xnorm, weights=w, zeta=BASE_ZETA)
        weight_rankings[name] = rank_from_grade(g)

   
    baseline_rank_df = load_baseline_ranking()
    compare_df = None
    if baseline_rank_df is not None:
        compare_df = ranking_df.merge(
            baseline_rank_df[["Alt_ID", "Rank"]],
            on="Alt_ID",
            how="left",
            suffixes=("_SmartDispatch", "_Baseline")
        )
        compare_df["RankShift_vs_Baseline"] = compare_df["Rank_SmartDispatch"] - compare_df["Rank_Baseline"]

   
    out_xlsx = os.path.join(OUT_DIR, f"GRA_SmartDispatch_WithSizeProxy_{stamp}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        ranking_df.to_excel(w, sheet_name="Ranking_Base", index=False)
        Xraw.reset_index().to_excel(w, sheet_name="DecisionMatrix_Used", index=False)
        Xnorm.reset_index().to_excel(w, sheet_name="Normalized_0to1", index=False)
        coeff_df.reset_index().to_excel(w, sheet_name="GreyCoeff_Base", index=False)

       
        zeta_rows = []
        for alt in ranking_df["Alt_ID"].tolist():
            row = {"Alt_ID": alt}
            for z in ZETA_SWEEP:
                row[f"Rank_zeta_{z}"] = zeta_rank_maps[z].get(alt, np.nan)
            zeta_rows.append(row)
        pd.DataFrame(zeta_rows).to_excel(w, sheet_name="ZetaSensitivity", index=False)

       
        for name, rdf in weight_rankings.items():
            safe_name = f"Rank_{name}"[:31]
            rdf.to_excel(w, sheet_name=safe_name, index=False)

        if compare_df is not None:
            compare_df.to_excel(w, sheet_name="Compare_vs_BaselineGRA", index=False)

        pd.DataFrame([{
            "SourceDispatchWorkbook": DISPATCH_SENS_XLSX,
            "SmartDMSheet": SMART_DM_SHEET,
            "AugmentedDMWorkbook": aug_path,
            "BaselineGRAWorkbook": BASELINE_GRA_XLSX,
            "RankMode": RANK_MODE,
            "BaseZeta": BASE_ZETA,
            "ZetaSweep": str(ZETA_SWEEP),
        }]).to_excel(w, sheet_name="RunInfo", index=False)

    
    topk_alts = ranking_df.head(TOPK)["Alt_ID"].tolist()
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    x = np.array(ZETA_SWEEP, dtype=float)
    for alt in topk_alts:
        y = [zeta_rank_maps[z].get(alt, np.nan) for z in ZETA_SWEEP]
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3.5, label=alt)
    ax.set_xlabel("ζ")
    ax.set_ylabel("Rank (lower is better)")
    ax.set_title(f"Smart-dispatch GRA ζ-sensitivity (Top {TOPK})")
    ax.invert_yaxis()
    ieee_axes_style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_SmartDispatch_ZetaSensitivity_Top{TOPK}_{stamp}"))

    print("\n✅ Smart-dispatch GRA DONE")
    print("Augmented smart DM:", aug_path)
    print("GRA output workbook:", out_xlsx)

    print("\nTop ranking from smart-dispatch GRA:")
    print(ranking_df.head(10).to_string(index=False))

    if compare_df is not None:
        print("\nComparison vs baseline GRA:")
        print(compare_df[["Alt_ID", "Rank_SmartDispatch", "Rank_Baseline", "RankShift_vs_Baseline"]]
              .sort_values("Rank_SmartDispatch")
              .to_string(index=False))


if __name__ == "__main__":
    main()