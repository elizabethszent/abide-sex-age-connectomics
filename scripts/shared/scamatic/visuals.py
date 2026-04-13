from pathlib import Path

# output folder
out_dir = Path("poster_ready_schematics")
out_dir.mkdir(exist_ok=True)

# --------------------------------------------------
# DX-only schematics
# --------------------------------------------------

# 1) Adult female | Z | M1 Somatomotor
make_dx_only_schematic(
    metric="Z",
    direction="decrease",   # CTL > ASD
    module_label="M1 Somatomotor",
    group_label="Adult female",
    title="Adult female | Z | M1 Somatomotor",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced within-module centrality in ASD",
    out_path=out_dir / "01_adult_female_Z_M1_dx_only.png",
)

# 2) Child female | Strength_neg | M2 Visual-A
make_dx_only_schematic(
    metric="Strength_neg",
    direction="decrease",   # CTL > ASD
    module_label="M2 Visual-A",
    group_label="Child female",
    title="Child female | Strength_neg | M2 Visual-A",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced negative strength in ASD",
    out_path=out_dir / "02_child_female_strengthneg_M2_dx_only.png",
)

# 3) Adult male | Strength_pos | M2 Visual-A
make_dx_only_schematic(
    metric="Strength_pos",
    direction="decrease",   # CTL > ASD
    module_label="M2 Visual-A",
    group_label="Adult male",
    title="Adult male | Strength_pos | M2 Visual-A",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced positive strength in ASD",
    out_path=out_dir / "03_adult_male_strengthpos_M2_dx_only.png",
)

# 4) Adult male | Strength_pos | M6 Visual-B
make_dx_only_schematic(
    metric="Strength_pos",
    direction="decrease",   # CTL > ASD
    module_label="M6 Visual-B",
    group_label="Adult male",
    title="Adult male | Strength_pos | M6 Visual-B",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced positive strength in ASD",
    out_path=out_dir / "04_adult_male_strengthpos_M6_dx_only.png",
)

# 5) Adult male | PC_pos | M4 Frontoparietal
make_dx_only_schematic(
    metric="PC_pos",
    direction="decrease",   # CTL > ASD
    module_label="M4 Frontoparietal",
    group_label="Adult male",
    title="Adult male | PC_pos | M4 Frontoparietal",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced positive cross-module participation in ASD",
    out_path=out_dir / "05_adult_male_PCpos_M4_dx_only.png",
)

# 6) Teen male | PC | M7 DefaultMode
make_dx_only_schematic(
    metric="PC",
    direction="decrease",   # CTL > ASD
    module_label="M7 DefaultMode",
    group_label="Teen male",
    title="Teen male | PC | M7 DefaultMode",
    stats_note="Primary threshold: FD < 0.3 | DX-only stream",
    footer_note="Interpretation: reduced cross-module participation in ASD",
    out_path=out_dir / "06_teen_male_PC_M7_dx_only.png",
)

# --------------------------------------------------
# DX×SEX schematics
# --------------------------------------------------

# 7) Adult | DX×SEX | Z | M1 Somatomotor
# Based on your interaction framing, this is a good first sex-heterogeneity figure.
make_dxsex_schematic(
    metric="Z",
    male_direction="decrease",
    female_direction="increase",
    module_label="M1 Somatomotor",
    age_label="Adult",
    title="Adult | DX×SEX | Z | M1 Somatomotor",
    stats_note="DX×SEX stream | compare d_male and d_female from your table",
    footer_note="Interpretation: diagnosis effect differs by sex",
    out_path=out_dir / "07_adult_Z_M1_dxsex.png",
)

# 8) Adult | DX×SEX | Z_pos | M1 Somatomotor
make_dxsex_schematic(
    metric="Z_pos",
    male_direction="decrease",
    female_direction="increase",
    module_label="M1 Somatomotor",
    age_label="Adult",
    title="Adult | DX×SEX | Z_pos | M1 Somatomotor",
    stats_note="DX×SEX stream | compare d_male and d_female from your table",
    footer_note="Interpretation: diagnosis effect differs by sex",
    out_path=out_dir / "08_adult_Zpos_M1_dxsex.png",
)

# 9) Child | DX×SEX | Strength_neg | M2 Visual-A
make_dxsex_schematic(
    metric="Strength_neg",
    male_direction="decrease",
    female_direction="decrease",
    module_label="M2 Visual-A",
    age_label="Child",
    title="Child | DX×SEX | Strength_neg | M2 Visual-A",
    stats_note="DX×SEX stream | stronger diagnosis effect in females",
    footer_note="Interpretation: same-direction diagnosis effect, larger in females",
    out_path=out_dir / "09_child_strengthneg_M2_dxsex.png",
)

# 10) Teen | DX×SEX | PC | M7 DefaultMode
make_dxsex_schematic(
    metric="PC",
    male_direction="decrease",
    female_direction="decrease",
    module_label="M7 DefaultMode",
    age_label="Teen",
    title="Teen | DX×SEX | PC | M7 DefaultMode",
    stats_note="DX×SEX stream | compare d_male and d_female from your table",
    footer_note="Interpretation: same-direction diagnosis effect with different magnitude by sex",
    out_path=out_dir / "10_teen_PC_M7_dxsex.png",
)