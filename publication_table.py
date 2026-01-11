import pandas as pd
import numpy as np
from scipy import stats
import os

# ============================================================
# CONFIGURATION - UPDATE BASE PATH IF NEEDED
# ============================================================

# Base directory (where 'data' and 'output' folders are located)
BASE_DIR = "."

# Input directories
DATA_DIR = os.path.join(BASE_DIR, "data")
PHASE3_DIR = os.path.join(BASE_DIR, "output", "Phase_3")
PHASE4_DIR = os.path.join(BASE_DIR, "output", "Phase_4")
PHASE5_DIR = os.path.join(BASE_DIR, "output", "Phase_5")
PHASE6_DIR = os.path.join(BASE_DIR, "output", "Phase_6")

# Input files - Discovery cohort
DISCOVERY_MATRIX = os.path.join(DATA_DIR, "GSE54564_series_matrix.txt")
DISCOVERY_ANNOTATION = os.path.join(DATA_DIR, "NCBI_Depression.bgx")

# Input files - Validation cohort
VALIDATION_EXPR = os.path.join(DATA_DIR, "GSE98793_prepared_expression.csv")
VALIDATION_META = os.path.join(DATA_DIR, "GSE98793_metadata.csv")
VALIDATION_FULL_META = os.path.join(DATA_DIR, "GSE98793_full_metadata.csv")

# Input files - Phase outputs
PHASE3_CELLTYPES = os.path.join(PHASE3_DIR, "Table_CellTypes.csv")
PHASE3_MODULES = os.path.join(PHASE3_DIR, "Table_Modules.csv")
PHASE4_DECONV = os.path.join(PHASE4_DIR, "GSE98793_deconvolution_results.csv")
PHASE4_STATS = os.path.join(PHASE4_DIR, "GSE98793_validation_stats.csv")
PHASE4_REPLICATION = os.path.join(PHASE4_DIR, "GSE98793_replication_summary.csv")
PHASE5_POS_GENES = os.path.join(PHASE5_DIR, "endothelial_correlated_positive.txt")
PHASE5_NEG_GENES = os.path.join(PHASE5_DIR, "endothelial_correlated_negative.txt")
PHASE5_ENRICHR_POS = os.path.join(PHASE5_DIR, "enrichr_positive_correlated.csv")
PHASE5_ENRICHR_NEG = os.path.join(PHASE5_DIR, "enrichr_negative_correlated.csv")
PHASE5_MODULE_ENDO = os.path.join(PHASE5_DIR, "module_endothelial_correlations.csv")
PHASE6_REGRESSION = os.path.join(PHASE6_DIR, "regression_summary.csv")
PHASE6_SUBGROUP = os.path.join(PHASE6_DIR, "effect_sizes_by_subgroup.csv")
PHASE6_ANXIETY = os.path.join(PHASE6_DIR, "anxiety_stratified_analysis.csv")

# Output directory - SEPARATE from figures
OUTPUT_DIR = os.path.join(BASE_DIR, "publication_output", "tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MARKER GENES DEFINITION
# ============================================================

MARKER_GENES = {
    'Neurons (Pan-neuronal)':{
        'markers':['SNAP25', 'SYT1', 'RBFOX3', 'MAP2', 'ENO2', 'SYN1', 'SYP', 'NRGN', 'NEFL'],
        'description':'Pan-neuronal markers'
    },
    'Neurons (Excitatory)':{
        'markers':['SLC17A7', 'SLC17A6', 'GRIN1', 'GRIN2A', 'GRIN2B', 'CAMK2A'],
        'description':'Glutamatergic neurons'
    },
    'Neurons (Inhibitory)':{
        'markers':['GAD1', 'GAD2', 'SLC32A1', 'PVALB', 'SST', 'VIP', 'NPY'],
        'description':'GABAergic neurons'
    },
    'Astrocytes':{
        'markers':['GFAP', 'AQP4', 'ALDH1L1', 'SLC1A2', 'SLC1A3', 'S100B', 'GLUL'],
        'description':'Astrocytic markers'
    },
    'Microglia':{
        'markers':['AIF1', 'CD68', 'ITGAM', 'CX3CR1', 'P2RY12', 'TMEM119', 'TREM2'],
        'description':'Microglial markers'
    },
    'Oligodendrocytes':{
        'markers':['MBP', 'MOG', 'PLP1', 'MAG', 'CNP', 'OLIG1', 'OLIG2'],
        'description':'Oligodendrocyte markers'
    },
    'Endothelial Cells':{
        'markers':['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'FLT1'],
        'description':'Vascular endothelial markers'
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_pvalue(p):
    """Format p-value for publication"""
    if pd.isna(p):
        return 'N/A'
    if p < 0.0001:
        return '<0.0001'
    elif p < 0.001:
        return f'{p:.4f}'
    elif p < 0.01:
        return f'{p:.3f}'
    else:
        return f'{p:.3f}'


# ============================================================
# TABLE 1: Cell-Type Marker Genes
# ============================================================

def create_table1_marker_genes():
    """
    Table 1: Brain Cell-Type Marker Gene Panels
    """
    print("Creating Table 1: Marker Genes...")

    rows = []
    for cell_type, info in MARKER_GENES.items():
        rows.append({
            'Cell Type':cell_type,
            'Description':info['description'],
            'Marker Genes':', '.join(info['markers']),
            'Number of Markers':len(info['markers'])
        })

    table1 = pd.DataFrame(rows)

    # Save as CSV
    table1.to_csv(f'{OUTPUT_DIR}/Table_1_Marker_Genes.csv', index=False)

    # Save as formatted text
    with open(f'{OUTPUT_DIR}/Table_1_Marker_Genes.txt', 'w') as f:
        f.write("TABLE 1. Brain Cell-Type Marker Gene Panels Used for Deconvolution\n")
        f.write("=" * 100 + "\n\n")
        f.write(table1.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Marker genes were selected based on established brain cell-type signatures.\n")
        f.write("Expression scores were calculated as the mean z-score of marker genes present in each dataset.\n")

    print(f"  Done - Table 1 saved ({len(rows)} cell types)")
    return table1


# ============================================================
# TABLE 2: Sample Characteristics
# ============================================================

def create_table2_sample_characteristics():
    """
    Table 2: Sample Characteristics of Discovery and Validation Cohorts
    """
    print("Creating Table 2: Sample Characteristics...")

    # Discovery cohort - extract from series matrix
    disc_n = 42
    disc_mdd = 21
    disc_ctrl = 21
    disc_age = "N/A"
    disc_female = "N/A"

    # Validation cohort - load metadata
    if os.path.exists(VALIDATION_META):
        val_meta = pd.read_csv(VALIDATION_META)
        val_n = len(val_meta)
        val_mdd = sum(val_meta['Status']=='MDD')
        val_ctrl = sum(val_meta['Status']=='Control')
    else:
        val_n = 192
        val_mdd = 128
        val_ctrl = 64

    # Try to get more details from full metadata
    val_age = "N/A"
    val_female = "N/A"
    val_anxiety = "N/A"

    if os.path.exists(VALIDATION_FULL_META):
        full_meta = pd.read_csv(VALIDATION_FULL_META, index_col=0)

        # Find age column
        age_cols = [c for c in full_meta.columns if 'age' in c.lower()]
        if age_cols:
            ages = pd.to_numeric(full_meta[age_cols[0]].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            val_age = f"{ages.mean():.1f} +/- {ages.std():.1f}"

        # Find sex column
        sex_cols = [c for c in full_meta.columns if 'gender' in c.lower() or 'sex' in c.lower()]
        if sex_cols:
            sex_data = full_meta[sex_cols[0]].astype(str).str.upper()
            n_female = sum(sex_data.str.contains('F'))
            pct_female = 100 * n_female / len(sex_data)
            val_female = f"{n_female} ({pct_female:.1f}%)"

        # Find anxiety column
        anx_cols = [c for c in full_meta.columns if 'anxiety' in c.lower()]
        if anx_cols:
            anx_data = full_meta[anx_cols[0]].astype(str).str.lower()
            n_anx = sum(anx_data.str.contains('yes|anxiety'))
            n_no_anx = sum(anx_data.str.contains('no'))
            val_anxiety = f"{n_anx} with anxiety, {n_no_anx} without"

    # Create table
    characteristics = [
        ('Total Samples', str(disc_n), str(val_n)),
        ('MDD Patients', str(disc_mdd), str(val_mdd)),
        ('Healthy Controls', str(disc_ctrl), str(val_ctrl)),
        ('Age (years), mean +/- SD', disc_age, val_age),
        ('Female, n (%)', disc_female, val_female),
        ('MDD Anxiety Status', 'N/A', val_anxiety),
        ('Platform', 'Illumina HumanHT-12 v4', 'Affymetrix U133 Plus 2.0'),
        ('Tissue', 'Whole Blood', 'Whole Blood'),
        ('GEO Accession', 'GSE54564', 'GSE98793'),
    ]

    table2 = pd.DataFrame(characteristics, columns=['Characteristic', 'Discovery Cohort', 'Validation Cohort'])

    # Save
    table2.to_csv(f'{OUTPUT_DIR}/Table_2_Sample_Characteristics.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_2_Sample_Characteristics.txt', 'w') as f:
        f.write("TABLE 2. Sample Characteristics of Discovery and Validation Cohorts\n")
        f.write("=" * 80 + "\n\n")
        f.write(table2.to_string(index=False))
        f.write("\n\n")
        f.write("Abbreviations: MDD, Major Depressive Disorder; SD, Standard Deviation; GEO, Gene Expression Omnibus\n")

    print(f"  Done - Table 2 saved")
    return table2


# ============================================================
# TABLE 3: Cell-Type Deconvolution Results
# ============================================================

def create_table3_deconvolution_results():
    """
    Table 3: Cell-Type Deconvolution Results Across Cohorts
    Uses pre-computed results from Phase 3 and Phase 4
    """
    print("Creating Table 3: Deconvolution Results...")

    # Load Phase 3 discovery results
    disc_results = pd.read_csv(PHASE3_CELLTYPES)

    # Load Phase 4 validation results
    val_results = pd.read_csv(PHASE4_STATS)

    # Load replication summary if available
    if os.path.exists(PHASE4_REPLICATION):
        replication = pd.read_csv(PHASE4_REPLICATION)
    else:
        replication = None

    # Merge results
    results = []

    # Get cell type column name from discovery
    ct_col_disc = disc_results.columns[0]

    for _, row in disc_results.iterrows():
        ct = row[ct_col_disc]

        # Get discovery stats
        d_disc = row.get('cohens_d', row.get('Cohens_d', row.get('Cohen_d', 0)))
        p_disc = row.get('p_value', row.get('P_value', row.get('p_mwu', 0)))

        # Find matching validation stats
        val_row = val_results[val_results['Cell_Type']==ct]
        if len(val_row) > 0:
            d_val = val_row['Cohens_d'].values[0]
            p_val = val_row['P_Value'].values[0]
        else:
            d_val = 0
            p_val = 1

        # Determine replication
        same_direction = np.sign(d_disc)==np.sign(d_val)
        replicated = 'Yes' if (same_direction and abs(d_disc) > 0.3 and p_val < 0.05) else 'No'

        results.append({
            'Cell Type':ct,
            'Discovery Cohen\'s d':f'{d_disc:.2f}',
            'Discovery P-value':format_pvalue(p_disc),
            'Validation Cohen\'s d':f'{d_val:.2f}',
            'Validation P-value':format_pvalue(p_val),
            'Direction Consistent':'Yes' if same_direction else 'No',
            'Replicated':replicated
        })

    table3 = pd.DataFrame(results)

    # Sort by absolute discovery effect size
    table3['sort_key'] = table3['Discovery Cohen\'s d'].astype(float).abs()
    table3 = table3.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

    # Save
    table3.to_csv(f'{OUTPUT_DIR}/Table_3_Deconvolution_Results.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_3_Deconvolution_Results.txt', 'w') as f:
        f.write("TABLE 3. Cell-Type Deconvolution Results: Discovery and Validation Cohorts\n")
        f.write("=" * 120 + "\n\n")
        f.write(table3.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Cohen's d represents standardized mean difference (MDD - Control).\n")
        f.write("Positive values indicate higher scores in MDD; negative values indicate lower scores in MDD.\n")
        f.write("Replication defined as: same direction, |d| > 0.3 in discovery, and p < 0.05 in validation.\n")
        f.write("P-values from Mann-Whitney U test (two-sided).\n")

    print(f"  Done - Table 3 saved ({len(results)} cell types)")
    return table3


# ============================================================
# TABLE 4: Pathway Enrichment Results
# ============================================================

def create_table4_pathway_enrichment():
    """
    Table 4: Pathway Enrichment Analysis of Endothelial-Correlated Genes
    """
    print("Creating Table 4: Pathway Enrichment...")

    # Load enrichment results
    enrichr_pos = pd.read_csv(PHASE5_ENRICHR_POS)
    enrichr_neg = pd.read_csv(PHASE5_ENRICHR_NEG)

    # Identify p-value column
    pval_col = [c for c in enrichr_pos.columns if 'p' in c.lower() and 'adj' in c.lower()]
    if not pval_col:
        pval_col = [c for c in enrichr_pos.columns if 'p' in c.lower()]
    pval_col = pval_col[0] if pval_col else 'adj_p_value'

    # Get top pathways
    pos_top = enrichr_pos.sort_values(pval_col).head(10).copy()
    neg_top = enrichr_neg.sort_values(pval_col).head(10).copy()

    pos_top['Direction'] = 'Positive'
    neg_top['Direction'] = 'Negative'

    # Combine
    combined = pd.concat([pos_top, neg_top], ignore_index=True)

    # Find column names
    term_col = [c for c in combined.columns if 'term' in c.lower()][0]
    overlap_col = [c for c in combined.columns if 'overlap' in c.lower() or 'genes' in c.lower()]
    overlap_col = overlap_col[0] if overlap_col else None
    or_col = [c for c in combined.columns if 'odds' in c.lower()]
    or_col = or_col[0] if or_col else None

    # Build table
    table_data = []
    for _, row in combined.iterrows():
        entry = {
            'Direction':row['Direction'],
            'Pathway':row[term_col][:60] + '...' if len(str(row[term_col])) > 60 else row[term_col],
            'Adjusted P-value':f'{row[pval_col]:.2e}'
        }
        if overlap_col:
            entry['Overlap'] = row[overlap_col]
        if or_col:
            entry['Odds Ratio'] = f'{row[or_col]:.2f}'
        table_data.append(entry)

    table4 = pd.DataFrame(table_data)

    # Save
    table4.to_csv(f'{OUTPUT_DIR}/Table_4_Pathway_Enrichment.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_4_Pathway_Enrichment.txt', 'w') as f:
        f.write("TABLE 4. Top Enriched Pathways Among Endothelial-Correlated Genes\n")
        f.write("=" * 120 + "\n\n")
        f.write(table4.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Pathway enrichment performed using Enrichr.\n")
        f.write("Positive direction: genes positively correlated with endothelial scores in both cohorts.\n")
        f.write("Negative direction: genes negatively correlated with endothelial scores in both cohorts.\n")
        f.write("P-values adjusted using Benjamini-Hochberg method.\n")

    print(f"  Done - Table 4 saved ({len(table4)} pathways)")
    return table4


# ============================================================
# TABLE 5: Regression Analysis Results
# ============================================================

def create_table5_regression_results():
    """
    Table 5: Multiple Regression Analysis of Endothelial Scores
    """
    print("Creating Table 5: Regression Results...")

    # Load regression results
    regression = pd.read_csv(PHASE6_REGRESSION)

    # Format table
    table5 = regression.copy()

    # Format p-values if present
    for col in table5.columns:
        if 'p' in col.lower() and table5[col].dtype in ['float64', 'float32']:
            table5[col] = table5[col].apply(lambda x:format_pvalue(x) if pd.notna(x) else 'N/A')

    # Save
    table5.to_csv(f'{OUTPUT_DIR}/Table_5_Regression_Results.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_5_Regression_Results.txt', 'w') as f:
        f.write("TABLE 5. Multiple Regression Analysis: Predictors of Endothelial Scores\n")
        f.write("=" * 100 + "\n\n")
        f.write(table5.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Analysis performed in validation cohort (GSE98793, n=192).\n")
        f.write("Model 1: Unadjusted (diagnosis only).\n")
        f.write("Model 2: Adjusted for age.\n")
        f.write("Model 3: Adjusted for age and anxiety status.\n")
        f.write("B = regression coefficient; CI = confidence interval.\n")

    print(f"  Done - Table 5 saved")
    return table5


# ============================================================
# TABLE 6: Subgroup Analysis
# ============================================================

def create_table6_subgroup_analysis():
    """
    Table 6: Subgroup Analysis by Anxiety Status
    """
    print("Creating Table 6: Subgroup Analysis...")

    # Load subgroup effects
    subgroup = pd.read_csv(PHASE6_SUBGROUP)

    # Format
    table6 = subgroup.copy()

    # Format p-values
    for col in table6.columns:
        if 'p' in col.lower() and table6[col].dtype in ['float64', 'float32']:
            table6[col] = table6[col].apply(lambda x:format_pvalue(x) if pd.notna(x) else 'N/A')

    # Save
    table6.to_csv(f'{OUTPUT_DIR}/Table_6_Subgroup_Analysis.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_6_Subgroup_Analysis.txt', 'w') as f:
        f.write("TABLE 6. Subgroup Analysis: Effect Sizes by Anxiety Status\n")
        f.write("=" * 100 + "\n\n")
        f.write(table6.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Analysis performed in validation cohort (GSE98793).\n")
        f.write("Cohen's d represents standardized mean difference (MDD subgroup - Control).\n")
        f.write("P-values from Mann-Whitney U test (two-sided).\n")

    print(f"  Done - Table 6 saved")
    return table6


# ============================================================
# SUPPLEMENTARY TABLE S1: Consensus Genes
# ============================================================

def create_table_s1_consensus_genes():
    """
    Supplementary Table S1: Endothelial-Correlated Consensus Genes
    """
    print("Creating Table S1: Consensus Genes...")

    # Load gene lists
    with open(PHASE5_POS_GENES, 'r') as f:
        pos_genes = [line.strip() for line in f if line.strip()]

    with open(PHASE5_NEG_GENES, 'r') as f:
        neg_genes = [line.strip() for line in f if line.strip()]

    # Create tables
    pos_df = pd.DataFrame({
        'Gene Symbol':pos_genes,
        'Correlation Direction':'Positive'
    })

    neg_df = pd.DataFrame({
        'Gene Symbol':neg_genes,
        'Correlation Direction':'Negative'
    })

    table_s1 = pd.concat([pos_df, neg_df], ignore_index=True)

    # Save
    table_s1.to_csv(f'{OUTPUT_DIR}/Table_S1_Consensus_Genes.csv', index=False)

    # Also save separate files
    pos_df.to_csv(f'{OUTPUT_DIR}/Table_S1a_Positive_Genes.csv', index=False)
    neg_df.to_csv(f'{OUTPUT_DIR}/Table_S1b_Negative_Genes.csv', index=False)

    with open(f'{OUTPUT_DIR}/Table_S1_Consensus_Genes.txt', 'w') as f:
        f.write("SUPPLEMENTARY TABLE S1. Endothelial-Correlated Consensus Genes\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total consensus genes: {len(table_s1)}\n")
        f.write(f"  - Positively correlated: {len(pos_genes)}\n")
        f.write(f"  - Negatively correlated: {len(neg_genes)}\n\n")
        f.write("Positively Correlated Genes:\n")
        f.write(", ".join(sorted(pos_genes)[:50]))
        if len(pos_genes) > 50:
            f.write(f"... and {len(pos_genes) - 50} more")
        f.write("\n\n")
        f.write("Negatively Correlated Genes:\n")
        f.write(", ".join(sorted(neg_genes)[:50]))
        if len(neg_genes) > 50:
            f.write(f"... and {len(neg_genes) - 50} more")
        f.write("\n\n")
        f.write("Note: Consensus genes showed significant correlation (FDR < 0.05)\n")
        f.write("with endothelial scores in both discovery and validation cohorts.\n")

    print(f"  Done - Table S1 saved ({len(pos_genes)} positive, {len(neg_genes)} negative)")
    return table_s1


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("MDD PUBLICATION TABLES GENERATOR")
    print("=" * 60)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\n" + "-" * 40)
    print("GENERATING TABLES")
    print("-" * 40 + "\n")

    # Main tables
    create_table1_marker_genes()
    create_table2_sample_characteristics()
    create_table3_deconvolution_results()
    create_table4_pathway_enrichment()
    create_table5_regression_results()
    create_table6_subgroup_analysis()

    # Supplementary tables
    create_table_s1_consensus_genes()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nAll tables saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__=="__main__":
    main()