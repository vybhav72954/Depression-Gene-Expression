#!/usr/bin/env python3
"""
MDD Endothelial Paper - Publication Tables
==========================================
Generates properly formatted tables for journal submission.

UPDATE THE FILE PATHS IN THE CONFIGURATION SECTION BEFORE RUNNING.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================

# Input files
DISCOVERY_DECONV = 'output/deconvolution_results.csv'
DISCOVERY_META = 'data/GSE54564_metadata.csv'
VALIDATION_DECONV = 'output/GSE98793_deconvolution_results.csv'
VALIDATION_META = 'data/GSE98793_metadata.csv'
ANALYSIS_META = 'data/GSE98793_analysis_metadata.csv'
ENRICHR_POSITIVE = 'output/enrichr_positive_correlated.csv'
ENRICHR_NEGATIVE = 'output/enrichr_negative_correlated.csv'
POSITIVE_GENES = 'output/endothelial_correlated_positive.txt'
NEGATIVE_GENES = 'output/endothelial_correlated_negative.txt'
REGRESSION_RESULTS = 'output/regression_summary.csv'
SUBGROUP_EFFECTS = 'output/effect_sizes_by_subgroup.csv'

# Output directory
OUTPUT_DIR = 'publication_tables'
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

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std==0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def format_pvalue(p):
    """Format p-value for publication"""
    if p < 0.0001:
        return '<0.0001'
    elif p < 0.001:
        return f'{p:.4f}'
    elif p < 0.01:
        return f'{p:.3f}'
    else:
        return f'{p:.3f}'


def standardize_diagnosis(df, status_col='Status'):
    """Standardize diagnosis column"""
    if status_col in df.columns:
        df['Diagnosis'] = df[status_col].apply(
            lambda x:'MDD' if any(s in str(x).upper() for s in ['MDD', 'CASE', 'PATIENT']) else 'Control'
        )
    return df


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

    print(f"  ✓ Table 1 saved ({len(rows)} cell types)")
    return table1


# ============================================================
# TABLE 2: Sample Characteristics
# ============================================================

def create_table2_sample_characteristics():
    """
    Table 2: Sample Characteristics of Discovery and Validation Cohorts
    """
    print("Creating Table 2: Sample Characteristics...")

    # Load metadata
    disc_meta = pd.read_csv(DISCOVERY_META)
    val_meta = pd.read_csv(VALIDATION_META)

    # Standardize
    disc_meta = standardize_diagnosis(disc_meta)
    val_meta = standardize_diagnosis(val_meta)

    # Discovery cohort stats
    disc_n = len(disc_meta)
    disc_mdd = sum(disc_meta['Diagnosis']=='MDD')
    disc_ctrl = sum(disc_meta['Diagnosis']=='Control')

    # Validation cohort stats
    val_n = len(val_meta)
    val_mdd = sum(val_meta['Diagnosis']=='MDD')
    val_ctrl = sum(val_meta['Diagnosis']=='Control')

    # Age if available
    disc_age = "N/A"
    val_age = "N/A"
    if 'Age' in disc_meta.columns:
        disc_age = f"{disc_meta['Age'].mean():.1f} ± {disc_meta['Age'].std():.1f}"
    if 'Age' in val_meta.columns:
        val_age = f"{val_meta['Age'].mean():.1f} ± {val_meta['Age'].std():.1f}"

    # Sex if available
    disc_female = "N/A"
    val_female = "N/A"
    if 'Sex' in disc_meta.columns:
        disc_female = f"{(disc_meta['Sex'].str.upper()=='F').sum()} ({100 * (disc_meta['Sex'].str.upper()=='F').mean():.1f}%)"
    if 'Sex' in val_meta.columns:
        val_female = f"{(val_meta['Sex'].str.upper()=='F').sum()} ({100 * (val_meta['Sex'].str.upper()=='F').mean():.1f}%)"

    # Anxiety if available (validation only)
    val_anxiety = "N/A"
    analysis_meta = pd.read_csv(ANALYSIS_META) if os.path.exists(ANALYSIS_META) else None
    if analysis_meta is not None and 'Anxiety' in analysis_meta.columns:
        mdd_data = analysis_meta[analysis_meta['Diagnosis']=='MDD']
        n_anx = sum(mdd_data['Anxiety']==1)
        n_no_anx = sum(mdd_data['Anxiety']==0)
        val_anxiety = f"{n_anx} with anxiety, {n_no_anx} without"

    # Create table
    characteristics = [
        ('Total Samples', str(disc_n), str(val_n)),
        ('MDD Patients', str(disc_mdd), str(val_mdd)),
        ('Healthy Controls', str(disc_ctrl), str(val_ctrl)),
        ('Age (years), mean ± SD', disc_age, val_age),
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

    print(f"  ✓ Table 2 saved")
    return table2


# ============================================================
# TABLE 3: Cell-Type Deconvolution Results
# ============================================================

def create_table3_deconvolution_results():
    """
    Table 3: Cell-Type Deconvolution Results Across Cohorts
    """
    print("Creating Table 3: Deconvolution Results...")

    # Load data
    disc_deconv = pd.read_csv(DISCOVERY_DECONV)
    disc_meta = pd.read_csv(DISCOVERY_META)
    val_deconv = pd.read_csv(VALIDATION_DECONV)
    val_meta = pd.read_csv(VALIDATION_META)

    # Merge
    disc_df = disc_deconv.merge(disc_meta, on='Sample', how='left')
    val_df = val_deconv.merge(val_meta, on='Sample', how='left')

    # Standardize diagnosis
    disc_df = standardize_diagnosis(disc_df)
    val_df = standardize_diagnosis(val_df)

    # Get cell types
    cell_types = [col for col in disc_deconv.columns if col not in ['Sample', 'Unnamed: 0']]

    results = []
    for ct in cell_types:
        if ct in disc_df.columns and ct in val_df.columns:
            # Discovery
            d_ctrl = disc_df[disc_df['Diagnosis']=='Control'][ct].dropna()
            d_mdd = disc_df[disc_df['Diagnosis']=='MDD'][ct].dropna()

            # Validation
            v_ctrl = val_df[val_df['Diagnosis']=='Control'][ct].dropna()
            v_mdd = val_df[val_df['Diagnosis']=='MDD'][ct].dropna()

            if len(d_ctrl) > 0 and len(d_mdd) > 0 and len(v_ctrl) > 0 and len(v_mdd) > 0:
                # Calculate statistics
                d_disc = cohens_d(d_mdd, d_ctrl)
                _, p_disc = stats.mannwhitneyu(d_mdd, d_ctrl, alternative='two-sided')

                d_val = cohens_d(v_mdd, v_ctrl)
                _, p_val = stats.mannwhitneyu(v_mdd, v_ctrl, alternative='two-sided')

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

    # Sort by discovery effect size
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

    print(f"  ✓ Table 3 saved ({len(results)} cell types)")
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
    enrichr_pos = pd.read_csv(ENRICHR_POSITIVE)
    enrichr_neg = pd.read_csv(ENRICHR_NEGATIVE)

    # Get top pathways
    pos_top = enrichr_pos.sort_values('Adjusted P-value').head(10).copy()
    neg_top = enrichr_neg.sort_values('Adjusted P-value').head(10).copy()

    pos_top['Correlation Direction'] = 'Positive'
    neg_top['Correlation Direction'] = 'Negative'

    # Combine
    combined = pd.concat([pos_top, neg_top], ignore_index=True)

    # Select and rename columns
    table4 = combined[['Correlation Direction', 'Term', 'Overlap', 'Adjusted P-value', 'Odds Ratio']].copy()
    table4.columns = ['Direction', 'Pathway', 'Overlap', 'Adjusted P-value', 'Odds Ratio']

    # Format values
    table4['Adjusted P-value'] = table4['Adjusted P-value'].apply(lambda x:f'{x:.2e}')
    table4['Odds Ratio'] = table4['Odds Ratio'].apply(lambda x:f'{x:.2f}')

    # Truncate long pathway names
    table4['Pathway'] = table4['Pathway'].apply(lambda x:x[:60] + '...' if len(x) > 60 else x)

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

    print(f"  ✓ Table 4 saved ({len(table4)} pathways)")
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
    regression = pd.read_csv(REGRESSION_RESULTS)

    # Format table
    table5 = regression.copy()

    # Standardize column names
    table5.columns = [col.replace('_', ' ').title() for col in table5.columns]

    # Format p-values if present
    for col in table5.columns:
        if 'p' in col.lower():
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
        f.write("β = standardized regression coefficient; CI = confidence interval.\n")

    print(f"  ✓ Table 5 saved")
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
    subgroup = pd.read_csv(SUBGROUP_EFFECTS)

    # Format
    table6 = subgroup.copy()

    # Format p-values
    for col in table6.columns:
        if 'p' in col.lower():
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

    print(f"  ✓ Table 6 saved")
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
    with open(POSITIVE_GENES, 'r') as f:
        pos_genes = [line.strip() for line in f if line.strip()]

    with open(NEGATIVE_GENES, 'r') as f:
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
        f.write(", ".join(pos_genes[:50]))
        if len(pos_genes) > 50:
            f.write(f"... and {len(pos_genes) - 50} more")
        f.write("\n\n")
        f.write("Negatively Correlated Genes:\n")
        f.write(", ".join(neg_genes[:50]))
        if len(neg_genes) > 50:
            f.write(f"... and {len(neg_genes) - 50} more")
        f.write("\n\n")
        f.write("Note: Consensus genes showed consistent correlation direction\n")
        f.write("(r > 0.25 or r < -0.25) with endothelial scores in both cohorts.\n")

    print(f"  ✓ Table S1 saved ({len(pos_genes)} positive, {len(neg_genes)} negative)")
    return table_s1


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("MDD PUBLICATION TABLES GENERATOR")
    print("=" * 60)

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
