import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime


OUT_DIR = r"C:\Users\Dell\Desktop\GRT\Codes\Results of stage 1"
OPTIONB_PATTERN = r"C:\Users\Dell\Desktop\GRT\Codes\Results of stage 1\Stage1_R1Slack_AllAlternatives_FINAL_OptionB_50scen_MeanWorst_20251217_212352.xlsx"
OPTIONB_SHEET = "DecisionMatrix_MeanWorst"

ALT_XLSX  = r"C:\Users\Dell\Desktop\GRT\Codes\Alternatives_CIGRE_R1toR18.xlsx"
ALT_SHEET = "Alternatives"


OUT_TAG = "PLUS_SizeProxy"


def find_latest(out_dir, pattern):
    files = sorted(glob.glob(os.path.join(out_dir, pattern)), key=os.path.getmtime)
    return files[-1] if files else None

def clean_alt_id(x):
    return str(x).strip()

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nFound: {df.columns.tolist()}")

def main():
    optionb_path = find_latest(OUT_DIR, OPTIONB_PATTERN)
    if optionb_path is None:
        raise FileNotFoundError(
            f"No Option-B file found in OUT_DIR:\n  {OUT_DIR}\nPattern: {OPTIONB_PATTERN}"
        )

    dm = pd.read_excel(optionb_path, sheet_name=OPTIONB_SHEET)
    if "Alt_ID" not in dm.columns:
        raise ValueError(f"'{OPTIONB_SHEET}' must contain 'Alt_ID'. Found: {dm.columns.tolist()}")
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
            "Some Alt_ID did not match Alternatives (merge failed). Check Alt_ID naming.\n"
            f"Unmatched Alt_ID examples:\n{bad.head(10).to_string(index=False)}"
        )

    
    out["InstalledCapacity_kW_mean"]  = out["InstalledCapacity_kW"]
    out["InstalledCapacity_kW_worst"] = out["InstalledCapacity_kW"]
    out["StorageEnergy_kWh_mean"]     = out["StorageEnergy_kWh"]
    out["StorageEnergy_kWh_worst"]    = out["StorageEnergy_kWh"]

    

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(optionb_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}_{OUT_TAG}_{stamp}.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="DecisionMatrix_MeanWorst", index=False)
        dm.to_excel(w, sheet_name="Original_DM", index=False)
        proxy.to_excel(w, sheet_name="SizeProxy_FromAlternatives", index=False)

    print("\n✅ Augmentation DONE")
    print("Input Option-B file:", optionb_path)
    print("Output augmented file:", out_path)
    print("Sheet to use for GRA: DecisionMatrix_MeanWorst")

if __name__ == "__main__":
    main()
