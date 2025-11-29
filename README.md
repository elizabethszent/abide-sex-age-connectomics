# Sex- and Age-Specific Functional Connectomics in Autism (ABIDE I)

This repository contains the analysis code for my CPSC 599 term project on **sex- and age-specific functional connectivity differences in Autism Spectrum Disorder (ASD)** using the **ABIDE I** dataset and the **CC200** functional atlas.

The project builds whole-brain functional networks, computes graph-theoretic measures (small-worldness, modularity, hub metrics), and compares **ASD vs. control** groups separately in **females vs. males** and across **child, teen, and young adult** age bands.

---

## 1. Project Goals

**High-level research questions**

1. **RQ1 – Development:**  
   Do ASD vs. control differences in **female functional connectivity** change with age?

2. **RQ2 – Network specificity:**  
   Are there specific **network pairs / edge clusters** that differ between ASD and control females?

3. **RQ3 – Modularity and hubs:**  
   Do females with ASD show differences in **modularity (Q)** and **hub roles**  
   (participation coefficient, within-module degree *z*) compared to controls?

**Short answer (from current results)**

- Networks in all groups show **strong small-world organization** and broadly similar large-scale topology.
- **Female ASD** shows **age-dependent, network-specific** connectivity differences  
  – broad ASD > control hyperconnectivity in childhood, more mixed patterns in teens, and subtle effects in young adults.
- **Male ASD** shows a more **stable pattern** with slightly reduced between-module connectivity from teen years onward.
- **Modularity and hub roles** (Q, PC, *z*) are **largely similar** between ASD and control females, with only small, non-robust differences in a few modules.

---

## 2. Repository Structure

The repo is organized around **data**, **analysis scripts**, and **results/figures**.

```text
CPSC_599_CONNECTOMICS/
├── data/
│   ├── connectomes/
│   │   └── cpac/nofilt_noglobal/cc200_z/
│   │       └── *.npy            # subject-level 200×200 FC matrices (Fisher z)
│   ├── female/
│   │   └── female_metadata_included.csv
│   └── male/
│       └── male_metadata_included.csv
│
├── results/
│   ├── group_connectomes/
│   │   └── CC200_modules.txt    # Louvain modules (ROI_index, Module 1..7)
│   ├── hubs/
│   │   ├── female_child_pc_z.csv
│   │   ├── female_teen_pc_z.csv
│   │   ├── female_young_adult_pc_z.csv
│   │   ├── male_child_pc_z.csv
│   │   ├── male_teen_pc_z.csv
│   │   ├── male_young_adult_pc_z.csv
│   │   ├── female_global_Q_by_age.csv
│   │   └── male_global_Q_by_age.csv
│   └── vis/
│       ├── brainnet/
│       │   ├── CC200_base.node
│       │   ├── CC200_louvain.node
│       │   └── nodes_by_sex_age/...   # node files for BrainNet Viewer
│       └── figures/
│           ├── CohendD_Female_*.png
│           ├── CohendD_Male_*.png
│           ├── female_Q_ASD_vs_CTL_by_age_boxplot.png
│           ├── male_Q_ASD_vs_CTL_by_age_boxplot.png
│           └── modularity_summary_panel.png
│
└── scripts/
    └── shared/
        ├── compute_pc_z_by_sex_age.py         # node-wise PC & z by sex/age
        ├── compute_global_Q_by_sex_age.py     # modularity Q per subject
        ├── plot_pc_z_effects_by_module.py     # Cohen's d for PC & z
        ├── plot_global_Q_by_sex_age.py        # boxplots + mean±SEM Q
        ├── make_CC200_louvain_node.py         # .node file with Louvain modules
        └── utils_*.py                         # shared helper functions
