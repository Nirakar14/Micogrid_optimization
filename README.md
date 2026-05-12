# Physics-Based Screening and Robust 
# Multi-Objective Optimization of Distributed 
# Energy Resource Architectures for LV Microgrids

## Paper Information
- **Journal:** Energy Conversion and Management: X
- **Authors:** Nirakar Nepal et al.
- **Institution:** Department of Electrical Engineering,
  IOE Pulchowk Campus, Tribhuvan University, Nepal
- **DOI:** [Paper](https://doi.org/10.1016/j.ecmx.2026.101933)

---

## Repository Contents

### Python Scripts (Code)
| File | Description |
|------|-------------|
| `simulator_wrapper.py` | Core simulation wrapper |
| `run_bfs_withoutDER.py` | Base case power flow |
| `run_gra_on_smart_dispatch_with_size_proxy.py` | Stage 1 GRA screening |
| `run_gra_optionB_with_size_proxy.py` | Stage 1 GRA  |
| `run_stage1_bfs_alternatives.py` | Stage 1 BFS alternatives |
| `stage1_dispatch_sensitivity.py` | Stage 1 sensitivity |
| `stage2_nsga_runner.py` | Stage 2 NSGA-II main runner |
| `stage2_nsga_runner_A13only.py` | Stage 2 A13 back-check |
| `stage2_nsga_runner_a14only.py` | Stage 2 A14 analysis |
| `stage2_nsga_runner_n20_convergence.py` | Convergence study |
| `stage2_repair_test_a1_a2.py` | A1/A2 repair test |
| `agument_optionB_decision_matrix_with_size_proxy.py` | Decision matrix |

### Data Files (Excel)
| File | Description |
|------|-------------|
| `Alternatives_CIGRE_R1toR18.xlsx` | DER architecture alternatives |
| `CIGRE_15day_Loads_R1toR18.xlsx` | 15-day load profiles (R1-R18) |
| `lineparameters.xlsx` | LV network line parameters |
| `PV_15day_Hourly_Profile_Khumaltar_NOCT.xlsx` | PV generation profiles |
| `Wind_15day_hourly_synthetic.xlsx` | Wind generation profiles |

---

## Requirements

### Python Version: 3.9 or higher

## How to Run

### Step 1: Clone the repository
### Step 2: Install dependencies
### Step 3: Update file paths
Open any runner script and update the paths
at the top of the file
### Step 4: Run Stage 1
### Step 5: Run Stage 2

## Network Description
- CIGRE European LV benchmark network
- 15-day scenario-based simulation
- DER technologies: PV, Wind, BESS, MT, FC

## Optimization Framework
- **Stage 1:** Physics-based screening using 
  Grey Relational Analysis (GRA)
- **Stage 2:** Robust multi-objective optimization 
  using NSGA-II 

---

## Citation
If you use this code or data, please cite: 
bibtex @article{NEPAL2026101933,
title = {Physics-based screening and robust multi-objective optimization of distributed energy resource architectures for LV microgrids},
journal = {Energy Conversion and Management: X},
volume = {31},
pages = {101933},
year = {2026},
issn = {2590-1745},
doi = {https://doi.org/10.1016/j.ecmx.2026.101933},
url = {https://www.sciencedirect.com/science/article/pii/S2590174526004162},
author = {Nirakar Nepal and Raagini Upadhyay and Anil Kumar Panjiyar},
keywords = {Distributed energy resources, Grey relational analysis, Multi-objective optimization, Robust planning under uncertainty},
abstract = {Low-voltage (LV) microgrids increasingly operate close to their technical limits as more distributed energy resources (DERs) and power-electronic controls are integrated at the distribution level. Planning must therefore respect network physics while balancing multiple objectives under uncertainty. This paper proposes a two-stage workflow that first screens a wide set of DER architecture options and then refines the best candidates using multi-objective optimization. Physics-based time-series simulations are used to evaluate each candidate, and Grey Relational Analysis is applied to rank alternatives. The top-ranked designs are then optimized using NSGA-II to obtain a trade-off between mean grid-import energy and total annual cost under worst-case voltage and operational constraints. Validation on unseen scenarios shows that some solutions that appear attractive during optimization do not remain feasible when conditions change. Among the shortlisted designs, the architecture with the highest DER penetration exhibits the strongest robustness. Overall, the workflow reduces computational effort while producing planning outcomes that remain reliable under uncertainty.}
}


---



## Contact
Nirakar Nepal
nirakarnepal71@gmail.com
Department of Electrical Engineering
IOE Pulchowk Campus, Tribhuvan University, Nepal
