import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


OUT_DIR = r"C:\Users\Dell\Desktop\GRT\Codes\Result_gra_after_size_cost_wrostremove"


AUG_PATTERN = r"C:\Users\Dell\Desktop\GRT\Codes\Results of stage 1\Stage1_R1Slack_AllAlternatives_FINAL_OptionB_50scen_MeanWorst_20251217_212352_PLUS_SizeProxy_20251218_074727_PLUS_SizeProxy_NoWorstDup_20251219_103911.xlsx"
SHEET_DM = "DecisionMatrix_MeanWorst"

RANK_MODE = "MEAN_WORST"  
BASE_ZETA = 0.5
ZETA_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]

N_RANDOM_WEIGHT_SETS = 500
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

==
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def find_latest(out_dir, pattern):
    files = sorted(glob.glob(os.path.join(out_dir, pattern)), key=os.path.getmtime)
    return files[-1] if files else None

def resolve_input_xlsx(maybe_path_or_pattern, fallback_dir):
    
    if isinstance(maybe_path_or_pattern, str) and os.path.isfile(maybe_path_or_pattern):
        return maybe_path_or_pattern
   
    x = find_latest(fallback_dir, maybe_path_or_pattern)
    return x

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
    fig.savefig(outbase + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")  # vector
    plt.close(fig)


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


def main():
    ensure_dir(OUT_DIR)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  
    xlsx_path = resolve_input_xlsx(AUG_PATTERN, OUT_DIR)
    if xlsx_path is None:
        raise FileNotFoundError(
            f"Could not find input file.\n"
            f"Given: {AUG_PATTERN}\n"
            f"If this is a pattern, make sure matching files exist inside:\n  {OUT_DIR}"
        )

    dm = pd.read_excel(xlsx_path, sheet_name=SHEET_DM)
    if "Alt_ID" not in dm.columns:
        raise ValueError(f"'{SHEET_DM}' must contain Alt_ID. Found: {dm.columns.tolist()}")

    dm = dm.copy()
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
        raise ValueError(
            "No criteria columns found ending with _mean/_worst.\n"
            f"Found columns: {dm.columns.tolist()}"
        )

    Xraw = dm[criteria_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    crit_types = [infer_metric_type(c) for c in criteria_cols]
    Xnorm = minmax_normalize(Xraw, crit_types)

 
    coeff_df, grade_s = gra_compute(Xnorm, weights=None, zeta=BASE_ZETA)
    ranking_df = rank_from_grade(grade_s)

 
    winner = ranking_df.iloc[0]["Alt_ID"]
    runner_up = ranking_df.iloc[1]["Alt_ID"] if len(ranking_df) > 1 else None
    has_a14 = "A14" in Xraw.index

    diag_lines = []
    diag_lines.append("=========== SANITY CHECKS (GRA) ===========")
    diag_lines.append(f"Input file: {xlsx_path}")
    diag_lines.append(f"Rank mode: {RANK_MODE}, zeta={BASE_ZETA}")
    diag_lines.append(f"Winner: {winner}  (Grade={float(grade_s.loc[winner]):.6f})")
    if runner_up is not None:
        diag_lines.append(f"Runner-up: {runner_up}  (Grade={float(grade_s.loc[runner_up]):.6f})")
    diag_lines.append("")

    def add_alt_compare(tag, alt_id):
        diag_lines.append(f"--- {tag}: {alt_id} ---")
        diag_lines.append("Raw criteria (sorted):")
        diag_lines.append(Xraw.loc[alt_id].sort_values().to_string())
        diag_lines.append("")
        diag_lines.append("Normalized criteria (sorted):")
        diag_lines.append(Xnorm.loc[alt_id].sort_values().to_string())
        diag_lines.append("")

    add_alt_compare("WINNER", winner)
    if runner_up is not None:
        add_alt_compare("RUNNER_UP", runner_up)
    if has_a14:
        add_alt_compare("REFERENCE (A14)", "A14")


    print("\n" + "\n".join(diag_lines) + "\n")

  
    diag_path = os.path.join(OUT_DIR, f"GRA_SanityDiagnostics_{stamp}.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(diag_lines))

   

    def short_crit_label(s):
       
        s = str(s)
        s = s.replace("_mean", "_m").replace("_worst", "_w")
        s = s.replace("InstalledCapacity_kW", "Cap_kW")
        s = s.replace("StorageEnergy_kWh", "Stor_kWh")
        s = s.replace("RenUtil_pct", "RenUtil_%")
        s = s.replace("Import_kWh", "GridImp_kWh")
        s = s.replace("Loss_kWh", "Loss_kWh")
        s = s.replace("BESS_cycles", "BESS_cyc")
        s = s.replace("Curt_kWh", "Curt_kWh")
        s = s.replace("Vmin_pu", "Vmin_pu")
        s = s.replace("ViolBH", "ViolBH")
        return s

    def short_weight_name(nm):
        
        mp = {
            "EqualWeights": "Equal",
            "VoltagePriority": "Volt",
            "RenewablePriority": "Ren",
            "LossPriority": "Loss",
            "GridIndependence": "Grid",
            "SizePenaltyStrong": "Size+",
            "SizePenaltyMild": "Size-",
        }
        return mp.get(nm, nm)

  
    grades = grade_s.sort_values(ascending=False)
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(len(grades)), grades.values)
    ax.set_xticks(np.arange(len(grades)))
    ax.set_xticklabels(grades.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Grey Relational Grade")
    ax.set_title("Grey Relational Grades")r
    ieee_axes_style(ax)

    
    for i in range(min(3, len(grades))):
        ax.text(i, grades.values[i] + 0.005, f"{grades.values[i]:.3f}", ha="center", va="bottom", fontsize=9)

    save_plot(fig, os.path.join(OUT_DIR, f"GRA_Grades_All_{stamp}"))

    
    topk_alts = ranking_df.head(TOPK)["Alt_ID"].tolist()
    Xtop = Xnorm.loc[topk_alts, :]

    
    mean_cols_here = [c for c in Xtop.columns if str(c).endswith("_mean")]
    worst_cols_here = [c for c in Xtop.columns if str(c).endswith("_worst")]

    if mean_cols_here:
        fig = plt.figure(figsize=(7.2, 4.0))
        ax = fig.add_subplot(111)
        x = np.arange(len(mean_cols_here))
        for alt in topk_alts:
            ax.plot(x, Xtop.loc[alt, mean_cols_here].values, marker="o", linewidth=1.2, markersize=3.5, label=alt)
        ax.set_xticks(x)
        ax.set_xticklabels([short_crit_label(c) for c in mean_cols_here], rotation=30, ha="right")
        ax.set_ylabel("Normalized value (0–1)")
        
        ieee_axes_style(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        save_plot(fig, os.path.join(OUT_DIR, f"GRA_Top{TOPK}_NormCriteria_MEAN_{stamp}"))

    if worst_cols_here:
        fig = plt.figure(figsize=(7.2, 4.0))
        ax = fig.add_subplot(111)
        x = np.arange(len(worst_cols_here))
        for alt in topk_alts:
            ax.plot(x, Xtop.loc[alt, worst_cols_here].values, marker="o", linewidth=1.2, markersize=3.5, label=alt)
        ax.set_xticks(x)
        ax.set_xticklabels([short_crit_label(c) for c in worst_cols_here], rotation=30, ha="right")
        ax.set_ylabel("Normalized value (0–1)")
        
        ieee_axes_style(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        save_plot(fig, os.path.join(OUT_DIR, f"GRA_Top{TOPK}_NormCriteria_WORST_{stamp}"))


    coeff_cols = list(coeff_df.columns)
    Ctop = coeff_df.loc[topk_alts, coeff_cols].values
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(Ctop, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(topk_alts)))
    ax.set_yticklabels(topk_alts)
    ax.set_xticks(np.arange(len(coeff_cols)))
    ax.set_xticklabels([short_crit_label(c) for c in coeff_cols], rotation=30, ha="right")
   
    ax.set_xlabel("Criteria")
    ax.set_ylabel("Alternative")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Grey coefficient")
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_Top{TOPK}_GreyCoeffHeatmap_{stamp}"))

    zeta_rank_maps = {}
    for z in [float(v) for v in ZETA_SWEEP]:
        _, g = gra_compute(Xnorm, weights=None, zeta=z)
        r = rank_from_grade(g)
        zeta_rank_maps[z] = r.set_index("Alt_ID")["Rank"].to_dict()

    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    for alt in topk_alts:
        ranks = [zeta_rank_maps[z].get(alt, np.nan) for z in ZETA_SWEEP]
        ax.plot(ZETA_SWEEP, ranks, marker="o", linewidth=1.2, markersize=3.5, label=alt)

    ax.set_xlabel("Distinguishing coefficient ζ")
    ax.set_ylabel("Rank (lower is better)")
    
    ax.invert_yaxis()
   
    ymin = int(np.nanmin([zeta_rank_maps[z].get(a, np.nan) for z in ZETA_SWEEP for a in topk_alts]))
    ymax = int(np.nanmax([zeta_rank_maps[z].get(a, np.nan) for z in ZETA_SWEEP for a in topk_alts]))
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ieee_axes_style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_ZetaSensitivity_Top{TOPK}_{stamp}"))

   
    weight_sets = build_weight_sets(criteria_cols)
    weight_rankings = {}
    for name, w in weight_sets.items():
        _, g = gra_compute(Xnorm, weights=w, zeta=BASE_ZETA)
        weight_rankings[name] = rank_from_grade(g)

    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    xnames_full = list(weight_sets.keys())
    xnames = [short_weight_name(nm) for nm in xnames_full]
    x = np.arange(len(xnames))

    for alt in topk_alts:
        y = []
        for nm in xnames_full:
            rmap = weight_rankings[nm].set_index("Alt_ID")["Rank"].to_dict()
            y.append(rmap.get(alt, np.nan))
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3.5, label=alt)

    ax.set_xticks(x)
    ax.set_xticklabels(xnames, rotation=20, ha="right")
    ax.set_ylabel("Rank (lower is better)")
  
    ax.invert_yaxis()
    ymin = int(np.nanmin([weight_rankings[nm].set_index("Alt_ID")["Rank"].get(a, np.nan) for nm in xnames_full for a in topk_alts]))
    ymax = int(np.nanmax([weight_rankings[nm].set_index("Alt_ID")["Rank"].get(a, np.nan) for nm in xnames_full for a in topk_alts]))
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ieee_axes_style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_WeightSetSensitivity_Top{TOPK}_{stamp}"))

 
    rng = np.random.default_rng(2025)
    ncrit = len(criteria_cols)
    alts = Xnorm.index.tolist()
    win = {a: 0 for a in alts}
    topk = {a: 0 for a in alts}

    for _ in range(int(N_RANDOM_WEIGHT_SETS)):
        w = rng.dirichlet(np.ones(ncrit))
        _, g = gra_compute(Xnorm, weights=w, zeta=BASE_ZETA)
        r = rank_from_grade(g)
        win[r.iloc[0]["Alt_ID"]] += 1
        for a in r.head(TOPK)["Alt_ID"].tolist():
            topk[a] += 1

    mc_df = pd.DataFrame({
        "Alt_ID": alts,
        "WinRate_rank1": [win[a] / N_RANDOM_WEIGHT_SETS for a in alts],
        f"Top{TOPK}_Rate": [topk[a] / N_RANDOM_WEIGHT_SETS for a in alts],
    }).sort_values("WinRate_rank1", ascending=False).reset_index(drop=True)

    top10 = mc_df.head(10).copy()

  
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    y = np.arange(len(top10))[::-1]
    ax.barh(y, top10["WinRate_rank1"].values[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(top10["Alt_ID"].values[::-1])
    ax.set_xlabel("Probability of Rank-1")
   
    ieee_axes_style(ax)
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_MC_WinRate_Top10_{stamp}"))

    
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    ax.barh(y, top10[f"Top{TOPK}_Rate"].values[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(top10["Alt_ID"].values[::-1])
    ax.set_xlabel(f"Probability of being in Top-{TOPK}")
    
    ieee_axes_style(ax)
    save_plot(fig, os.path.join(OUT_DIR, f"GRA_MC_Top{TOPK}Rate_Top10_{stamp}"))


  
    out_xlsx = os.path.join(OUT_DIR, f"GRA_Ranking_SizeProxy_{stamp}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        Xraw.reset_index().to_excel(w, sheet_name="DecisionMatrix_Used", index=False)
        Xnorm.reset_index().to_excel(w, sheet_name="Normalized_0to1", index=False)
        coeff_df.reset_index().to_excel(w, sheet_name="GreyCoeff_Base", index=False)
        ranking_df.to_excel(w, sheet_name="Ranking_Base", index=False)
        mc_df.to_excel(w, sheet_name="MonteCarlo_WeightStab", index=False)

        for name, r in weight_rankings.items():
            r.to_excel(w, sheet_name=("Rank_" + name)[:31], index=False)

        meta = pd.DataFrame({
            "Item": [
                "Input augmented file",
                "Input sheet",
                "Rank mode",
                "Base zeta",
                "Zeta sweep",
                "Random weight samples",
                "TopK used",
                "Criteria columns",
                "Sanity diagnostics file"
            ],
            "Value": [
                xlsx_path,
                SHEET_DM,
                RANK_MODE,
                BASE_ZETA,
                str(ZETA_SWEEP),
                N_RANDOM_WEIGHT_SETS,
                TOPK,
                ", ".join(criteria_cols),
                diag_path
            ]
        })
        meta.to_excel(w, sheet_name="RunInfo", index=False)

    print("\n================== GRA DONE (Size Proxy) ==================")
    print("Input augmented file:", xlsx_path)
    print("Sanity diagnostics saved:", diag_path)
    print("Output Excel:", out_xlsx)
    print("\nTop 10 ranking:")
    print(ranking_df.head(10).to_string(index=False))
    print("\nPlots saved in OUT_DIR:", OUT_DIR)
    print("===========================================================\n")

if __name__ == "__main__":
    main()
