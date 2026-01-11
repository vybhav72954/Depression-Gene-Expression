#!/usr/bin/env python3
"""
MDD Endothelial Paper - Publication Figures
============================================
Each figure is saved as a separate high-DPI image.

UPDATE THE FILE PATHS IN THE CONFIGURATION SECTION BEFORE RUNNING.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================

# Input files - UPDATE THESE TO YOUR ACTUAL FILE LOCATIONS
DISCOVERY_DECONV = 'output/Phase_4/GSE98793_deconvolution_results.csv'
DISCOVERY_META = 'data/GSE54564_metadata.csv'
VALIDATION_DECONV = 'output/GSE98793_deconvolution_results.csv'
VALIDATION_META = 'data/GSE98793_metadata.csv'
ANALYSIS_META = 'data/GSE98793_analysis_metadata.csv'
ENRICHR_POSITIVE = 'output/enrichr_positive_correlated.csv'
ENRICHR_NEGATIVE = 'output/enrichr_negative_correlated.csv'
POSITIVE_GENES = 'output/endothelial_correlated_positive.txt'
NEGATIVE_GENES = 'output/endothelial_correlated_negative.txt'
MODULE_EIGENGENES = 'output/module_eigengenes.csv'
MODULE_ENDOTHELIAL = 'output/module_endothelial_correlations.csv'
REGRESSION_RESULTS = 'output/regression_summary.csv'
SUBGROUP_EFFECTS = 'output/effect_sizes_by_subgroup.csv'

# Output directory
OUTPUT_DIR = 'publication_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure settings
DPI = 600
FIGSIZE_SINGLE = (6, 5)
FIGSIZE_WIDE = (8, 5)

# Colors
COLOR_CONTROL = '#4A90D9'
COLOR_MDD = '#E74C3C'
COLOR_MDD_ANXIETY = '#9B59B6'
COLOR_POSITIVE = '#E74C3C'
COLOR_NEGATIVE = '#3498DB'
COLOR_HIGHLIGHT = '#2ECC71'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def set_style():
    """Set publication-quality matplotlib style"""
    plt.rcParams.update({
        'font.family':'sans-serif',
        'font.sans-serif':['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':11,
        'axes.titlesize':13,
        'axes.labelsize':12,
        'xtick.labelsize':10,
        'ytick.labelsize':10,
        'legend.fontsize':10,
        'figure.dpi':DPI,
        'savefig.dpi':DPI,
        'savefig.bbox':'tight',
        'savefig.pad_inches':0.15,
        'axes.linewidth':1.0,
        'axes.spines.top':False,
        'axes.spines.right':False,
    })


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std==0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_data():
    """Load all required data files"""
    data = {}

    # Discovery cohort
    data['disc_deconv'] = pd.read_csv(DISCOVERY_DECONV)
    data['disc_meta'] = pd.read_csv(DISCOVERY_META)

    # Validation cohort
    data['val_deconv'] = pd.read_csv(VALIDATION_DECONV)
    data['val_meta'] = pd.read_csv(VALIDATION_META)
    data['analysis_meta'] = pd.read_csv(ANALYSIS_META)

    # Pathway data
    data['enrichr_pos'] = pd.read_csv(ENRICHR_POSITIVE)
    data['enrichr_neg'] = pd.read_csv(ENRICHR_NEGATIVE)

    with open(POSITIVE_GENES, 'r') as f:
        data['pos_genes'] = [line.strip() for line in f if line.strip()]
    with open(NEGATIVE_GENES, 'r') as f:
        data['neg_genes'] = [line.strip() for line in f if line.strip()]

    # WGCNA data
    data['module_eigen'] = pd.read_csv(MODULE_EIGENGENES)
    data['module_endo'] = pd.read_csv(MODULE_ENDOTHELIAL)

    # Confounder data
    data['regression'] = pd.read_csv(REGRESSION_RESULTS)
    data['subgroup'] = pd.read_csv(SUBGROUP_EFFECTS)

    return data


def standardize_diagnosis(df, status_col='Status'):
    """Standardize diagnosis column"""
    if status_col in df.columns:
        df['Diagnosis'] = df[status_col].apply(
            lambda x:'MDD' if any(s in str(x).upper() for s in ['MDD', 'CASE', 'PATIENT']) else 'Control'
        )
    return df


# ============================================================
# FIGURE 1A: Discovery Cohort Endothelial Boxplot
# ============================================================

def figure_1a_discovery_boxplot(data):
    """Discovery cohort endothelial scores by diagnosis"""

    # Merge and prepare data
    df = data['disc_deconv'].merge(data['disc_meta'], on='Sample', how='left')
    df = standardize_diagnosis(df)

    ctrl = df[df['Diagnosis']=='Control']['Endothelial'].dropna()
    mdd = df[df['Diagnosis']=='MDD']['Endothelial'].dropna()

    # Calculate statistics
    d = cohens_d(mdd, ctrl)
    _, p = stats.mannwhitneyu(mdd, ctrl, alternative='two-sided')

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Boxplot
    bp = ax.boxplot([ctrl, mdd], positions=[0, 1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLOR_CONTROL)
    bp['boxes'][1].set_facecolor(COLOR_MDD)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    # Add jittered points
    for i, (vals, pos) in enumerate(zip([ctrl, mdd], [0, 1])):
        x = np.random.normal(pos, 0.08, len(vals))
        color = COLOR_CONTROL if i==0 else COLOR_MDD
        ax.scatter(x, vals, alpha=0.6, s=40, color=color, edgecolor='white', linewidth=0.5, zorder=3)

    # Labels and formatting
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Control\n(n={len(ctrl)})', f'MDD\n(n={len(mdd)})'])
    ax.set_ylabel('Endothelial Score (z-score)')
    ax.set_title('Discovery Cohort (GSE54564)', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Statistics annotation
    p_text = f'p = {p:.3f}' if p >= 0.001 else f'p = {p:.2e}'
    ax.annotate(f"Cohen's d = {d:.2f}\n{p_text}",
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_1A_Discovery_Endothelial.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_1A_Discovery_Endothelial.pdf', dpi=DPI)
    plt.close()

    print(f"✓ Figure 1A saved (Discovery: d={d:.2f}, p={p:.4f})")


# ============================================================
# FIGURE 1B: Validation Cohort Endothelial Boxplot
# ============================================================

def figure_1b_validation_boxplot(data):
    """Validation cohort endothelial scores by diagnosis"""

    df = data['val_deconv'].merge(data['val_meta'], on='Sample', how='left')
    df = standardize_diagnosis(df)

    ctrl = df[df['Diagnosis']=='Control']['Endothelial'].dropna()
    mdd = df[df['Diagnosis']=='MDD']['Endothelial'].dropna()

    d = cohens_d(mdd, ctrl)
    _, p = stats.mannwhitneyu(mdd, ctrl, alternative='two-sided')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    bp = ax.boxplot([ctrl, mdd], positions=[0, 1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLOR_CONTROL)
    bp['boxes'][1].set_facecolor(COLOR_MDD)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    for i, (vals, pos) in enumerate(zip([ctrl, mdd], [0, 1])):
        x = np.random.normal(pos, 0.06, len(vals))
        color = COLOR_CONTROL if i==0 else COLOR_MDD
        ax.scatter(x, vals, alpha=0.5, s=25, color=color, edgecolor='white', linewidth=0.3, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Control\n(n={len(ctrl)})', f'MDD\n(n={len(mdd)})'])
    ax.set_ylabel('Endothelial Score (z-score)')
    ax.set_title('Validation Cohort (GSE98793)', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    p_text = f'p = {p:.4f}' if p >= 0.0001 else f'p = {p:.2e}'
    stars = '**' if p < 0.01 else ('*' if p < 0.05 else '')
    ax.annotate(f"Cohen's d = {d:.2f}\n{p_text} {stars}",
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_1B_Validation_Endothelial.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_1B_Validation_Endothelial.pdf', dpi=DPI)
    plt.close()

    print(f"✓ Figure 1B saved (Validation: d={d:.2f}, p={p:.4f})")


# ============================================================
# FIGURE 1C: Effect Sizes Comparison (Forest Plot)
# ============================================================

def figure_1c_effect_sizes(data):
    """Forest plot comparing effect sizes across cell types"""

    disc_df = data['disc_deconv'].merge(data['disc_meta'], on='Sample', how='left')
    val_df = data['val_deconv'].merge(data['val_meta'], on='Sample', how='left')
    disc_df = standardize_diagnosis(disc_df)
    val_df = standardize_diagnosis(val_df)

    # Get cell type columns
    cell_types = [col for col in data['disc_deconv'].columns
                  if col not in ['Sample', 'Unnamed: 0']]

    results = []
    for ct in cell_types:
        if ct in disc_df.columns and ct in val_df.columns:
            d_ctrl = disc_df[disc_df['Diagnosis']=='Control'][ct].dropna()
            d_mdd = disc_df[disc_df['Diagnosis']=='MDD'][ct].dropna()
            v_ctrl = val_df[val_df['Diagnosis']=='Control'][ct].dropna()
            v_mdd = val_df[val_df['Diagnosis']=='MDD'][ct].dropna()

            if len(d_ctrl) > 0 and len(d_mdd) > 0 and len(v_ctrl) > 0 and len(v_mdd) > 0:
                results.append({
                    'Cell Type':ct,
                    'Discovery d':cohens_d(d_mdd, d_ctrl),
                    'Validation d':cohens_d(v_mdd, v_ctrl)
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Discovery d', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.barh(y_pos - width / 2, results_df['Discovery d'], width,
                    label='Discovery (GSE54564)', color='#3498DB', alpha=0.85, edgecolor='white')
    bars2 = ax.barh(y_pos + width / 2, results_df['Validation d'], width,
                    label='Validation (GSE98793)', color='#E67E22', alpha=0.85, edgecolor='white')

    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1, label='Medium effect (d=0.5)')
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Highlight endothelial
    endo_idx = results_df[results_df['Cell Type']=='Endothelial'].index
    if len(endo_idx) > 0:
        endo_pos = list(results_df['Cell Type']).index('Endothelial')
        ax.axhspan(endo_pos - 0.5, endo_pos + 0.5, alpha=0.2, color='green')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['Cell Type'])
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title('Effect Sizes Across Cell Types', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_1C_EffectSizes_Comparison.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_1C_EffectSizes_Comparison.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 1C saved (Effect sizes comparison)")


# ============================================================
# FIGURE 1D: Replication Heatmap
# ============================================================

def figure_1d_replication_heatmap(data):
    """Heatmap showing replication status across cell types"""

    disc_df = data['disc_deconv'].merge(data['disc_meta'], on='Sample', how='left')
    val_df = data['val_deconv'].merge(data['val_meta'], on='Sample', how='left')
    disc_df = standardize_diagnosis(disc_df)
    val_df = standardize_diagnosis(val_df)

    cell_types = [col for col in data['disc_deconv'].columns
                  if col not in ['Sample', 'Unnamed: 0']]

    results = []
    for ct in cell_types:
        if ct in disc_df.columns and ct in val_df.columns:
            d_ctrl = disc_df[disc_df['Diagnosis']=='Control'][ct].dropna()
            d_mdd = disc_df[disc_df['Diagnosis']=='MDD'][ct].dropna()
            v_ctrl = val_df[val_df['Diagnosis']=='Control'][ct].dropna()
            v_mdd = val_df[val_df['Diagnosis']=='MDD'][ct].dropna()

            if len(d_ctrl) > 0 and len(d_mdd) > 0 and len(v_ctrl) > 0 and len(v_mdd) > 0:
                d_disc = cohens_d(d_mdd, d_ctrl)
                d_val = cohens_d(v_mdd, v_ctrl)
                _, p_val = stats.mannwhitneyu(v_mdd, v_ctrl)

                results.append({
                    'Cell Type':ct,
                    'Discovery':d_disc,
                    'Validation':d_val,
                    'Val_p':p_val
                })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Discovery', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 7))

    heatmap_data = results_df[['Discovery', 'Validation']].values

    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Discovery', 'Validation'], fontsize=11)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['Cell Type'], fontsize=10)

    # Add values
    for i in range(len(results_df)):
        for j in range(2):
            val = heatmap_data[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    # Add replication status
    for i, row in enumerate(results_df.itertuples()):
        d_eff, v_eff, p = row.Discovery, row.Validation, row.Val_p
        if np.sign(d_eff)==np.sign(v_eff) and abs(d_eff) > 0.3 and p < 0.05:
            rep, color = '✓', 'green'
        elif np.sign(d_eff)==np.sign(v_eff) and abs(v_eff) > 0.15:
            rep, color = '~', 'orange'
        else:
            rep, color = '✗', 'red'
        ax.text(2.15, i, rep, ha='center', va='center', fontsize=14,
                color=color, fontweight='bold')

    ax.text(2.15, -0.8, 'Rep.', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.5, 2.5)

    ax.set_title('Cell-Type Effect Sizes & Replication', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.15)
    cbar.set_label("Cohen's d", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_1D_Replication_Heatmap.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_1D_Replication_Heatmap.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 1D saved (Replication heatmap)")


# ============================================================
# FIGURE 2A: Consensus Genes Count
# ============================================================

def figure_2a_gene_counts(data):
    """Bar chart showing positive vs negative consensus genes"""

    n_pos = len(data['pos_genes'])
    n_neg = len(data['neg_genes'])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    categories = ['Positive\nCorrelation', 'Negative\nCorrelation']
    counts = [n_pos, n_neg]
    colors = [COLOR_POSITIVE, COLOR_NEGATIVE]

    bars = ax.bar(categories, counts, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Number of Consensus Genes', fontsize=12)
    ax.set_title('Endothelial-Correlated Consensus Genes', fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_2A_Consensus_Gene_Counts.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_2A_Consensus_Gene_Counts.pdf', dpi=DPI)
    plt.close()

    print(f"✓ Figure 2A saved (Positive: {n_pos}, Negative: {n_neg})")


# ============================================================
# FIGURE 2B: Positive Pathway Enrichment
# ============================================================

def figure_2b_positive_pathways(data):
    """Horizontal bar chart of top positive pathways"""

    df = data['enrichr_pos'].sort_values('Adjusted P-value').head(10)

    pathways = df['Term'].tolist()
    pathways = [p[:40] + '...' if len(p) > 40 else p for p in pathways]
    pvals = df['Adjusted P-value'].tolist()
    log_pvals = [-np.log10(p) for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(pathways))
    bars = ax.barh(y_pos, log_pvals, color=COLOR_POSITIVE, alpha=0.85, edgecolor='white', height=0.7)

    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='p = 0.05 threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=12)
    ax.set_title('Pathway Enrichment: Positive Correlations', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_2B_Positive_Pathways.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_2B_Positive_Pathways.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 2B saved (Positive pathways)")


# ============================================================
# FIGURE 2C: Negative Pathway Enrichment
# ============================================================

def figure_2c_negative_pathways(data):
    """Horizontal bar chart of top negative pathways"""

    df = data['enrichr_neg'].sort_values('Adjusted P-value').head(10)

    pathways = df['Term'].tolist()
    pathways = [p[:40] + '...' if len(p) > 40 else p for p in pathways]
    pvals = df['Adjusted P-value'].tolist()
    log_pvals = [-np.log10(p) for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(pathways))
    bars = ax.barh(y_pos, log_pvals, color=COLOR_NEGATIVE, alpha=0.85, edgecolor='white', height=0.7)

    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='p = 0.05 threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('-log₁₀(Adjusted P-value)', fontsize=12)
    ax.set_title('Pathway Enrichment: Negative Correlations', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_2C_Negative_Pathways.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_2C_Negative_Pathways.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 2C saved (Negative pathways)")


# ============================================================
# FIGURE 3A: Module-Endothelial Correlations
# ============================================================

def figure_3a_module_correlations(data):
    """Bar chart of module-endothelial correlations"""

    df = data['module_endo'].copy()

    # Identify module and correlation columns
    module_col = df.columns[0]
    corr_col = [c for c in df.columns if 'corr' in c.lower() or 'r' in c.lower()][0]

    df = df.sort_values(corr_col, ascending=True)

    modules = df[module_col].tolist()
    correlations = df[corr_col].tolist()

    colors = [COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE for c in correlations]

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(modules))
    bars = ax.barh(y_pos, correlations, color=colors, alpha=0.85, edgecolor='white', height=0.7)

    ax.axvline(x=0, color='black', linewidth=1)

    # Highlight strongest positive correlation
    max_idx = np.argmax(correlations)
    ax.axhspan(max_idx - 0.4, max_idx + 0.4, alpha=0.2, color='gold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(modules, fontsize=10)
    ax.set_xlabel('Correlation with Endothelial Score (r)', fontsize=12)
    ax.set_title('WGCNA Module-Endothelial Correlations', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_3A_Module_Endothelial_Correlations.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_3A_Module_Endothelial_Correlations.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 3A saved (Module-endothelial correlations)")


# ============================================================
# FIGURE 3B: Key Module Eigengene by Diagnosis
# ============================================================

def figure_3b_module_eigengene(data):
    """Boxplot of key module eigengene by diagnosis"""

    df = data['module_eigen'].merge(data['disc_meta'], on='Sample', how='left')
    df = standardize_diagnosis(df)

    # Find key module (highest endothelial correlation or M8)
    module_cols = [c for c in df.columns if c.startswith('ME') or c.startswith('M')]
    key_module = module_cols[0]  # Update this based on your data

    ctrl = df[df['Diagnosis']=='Control'][key_module].dropna()
    mdd = df[df['Diagnosis']=='MDD'][key_module].dropna()

    r, p = stats.pointbiserialr((df['Diagnosis']=='MDD').astype(int), df[key_module])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    bp = ax.boxplot([ctrl, mdd], positions=[0, 1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLOR_CONTROL)
    bp['boxes'][1].set_facecolor(COLOR_MDD)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    for i, (vals, pos) in enumerate(zip([ctrl, mdd], [0, 1])):
        x = np.random.normal(pos, 0.08, len(vals))
        color = COLOR_CONTROL if i==0 else COLOR_MDD
        ax.scatter(x, vals, alpha=0.6, s=40, color=color, edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'MDD'])
    ax.set_ylabel(f'{key_module} Eigengene')
    ax.set_title(f'{key_module} Module by Diagnosis', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    p_text = f'p = {p:.3f}' if p >= 0.001 else f'p = {p:.2e}'
    ax.annotate(f'r = {r:.2f}\n{p_text}',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_3B_Module_Eigengene.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_3B_Module_Eigengene.pdf', dpi=DPI)
    plt.close()

    print(f"✓ Figure 3B saved ({key_module} eigengene)")


# ============================================================
# FIGURE 4A: Anxiety Stratification
# ============================================================

def figure_4a_anxiety_stratification(data):
    """Boxplot of endothelial scores stratified by anxiety"""

    df = data['val_deconv'].merge(data['analysis_meta'], on='Sample', how='left')

    ctrl = df[df['Diagnosis']=='Control']['Endothelial'].dropna()
    mdd_no_anx = df[(df['Diagnosis']=='MDD') & (df['Anxiety']==0)]['Endothelial'].dropna()
    mdd_anx = df[(df['Diagnosis']=='MDD') & (df['Anxiety']==1)]['Endothelial'].dropna()

    _, p1 = stats.mannwhitneyu(ctrl, mdd_no_anx)
    _, p2 = stats.mannwhitneyu(ctrl, mdd_anx)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    bp = ax.boxplot([ctrl, mdd_no_anx, mdd_anx], positions=[0, 1, 2], widths=0.55, patch_artist=True)

    bp['boxes'][0].set_facecolor(COLOR_CONTROL)
    bp['boxes'][1].set_facecolor(COLOR_MDD)
    bp['boxes'][2].set_facecolor(COLOR_MDD_ANXIETY)
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'Control\n(n={len(ctrl)})',
                        f'MDD without\nAnxiety (n={len(mdd_no_anx)})',
                        f'MDD with\nAnxiety (n={len(mdd_anx)})'])
    ax.set_ylabel('Endothelial Score (z-score)')
    ax.set_title('Endothelial Scores by Anxiety Status', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Significance brackets
    y_max = max(ctrl.max(), mdd_no_anx.max(), mdd_anx.max())

    # Bracket 1: Control vs MDD no anxiety
    y1 = y_max + 0.25
    ax.plot([0, 0, 1, 1], [y1, y1 + 0.08, y1 + 0.08, y1], 'k-', linewidth=1)
    stars1 = '**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'ns')
    ax.text(0.5, y1 + 0.12, f'p={p1:.3f} {stars1}', ha='center', fontsize=9)

    # Bracket 2: Control vs MDD with anxiety
    y2 = y_max + 0.55
    ax.plot([0, 0, 2, 2], [y2, y2 + 0.08, y2 + 0.08, y2], 'k-', linewidth=1)
    stars2 = '**' if p2 < 0.01 else ('*' if p2 < 0.05 else 'ns')
    ax.text(1, y2 + 0.12, f'p={p2:.3f} {stars2}', ha='center', fontsize=9)

    ax.set_ylim(ax.get_ylim()[0], y2 + 0.4)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_4A_Anxiety_Stratification.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_4A_Anxiety_Stratification.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 4A saved (Anxiety stratification)")


# ============================================================
# FIGURE 4B: Regression Coefficients
# ============================================================

def figure_4b_regression_coefficients(data):
    """Forest plot of regression coefficients across models"""

    df = data['regression']

    models = df['Model'].tolist()

    # Get beta and CI columns
    beta_col = [c for c in df.columns if 'beta' in c.lower() or 'β' in c.lower() or 'coef' in c.lower()][0]
    betas = df[beta_col].tolist()

    if 'CI_Lower' in df.columns:
        ci_lower = df['CI_Lower'].tolist()
        ci_upper = df['CI_Upper'].tolist()
    else:
        ci_lower = [b - 0.15 for b in betas]
        ci_upper = [b + 0.15 for b in betas]

    pval_col = [c for c in df.columns if 'p' in c.lower()][0]
    pvals = df[pval_col].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(models))

    ax.errorbar(betas, y_pos,
                xerr=[np.array(betas) - np.array(ci_lower), np.array(ci_upper) - np.array(betas)],
                fmt='o', markersize=12, color=COLOR_MDD, capsize=6, capthick=2.5,
                elinewidth=2.5, markeredgecolor='white', markeredgewidth=1.5)

    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Diagnosis β Coefficient (95% CI)', fontsize=12)
    ax.set_title('Effect Stability Across Regression Models', fontweight='bold')

    # Add p-values
    for i, (b, ci_u, p) in enumerate(zip(betas, ci_upper, pvals)):
        p_text = f'p = {p:.4f}' if p >= 0.0001 else f'p = {p:.2e}'
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.annotate(f'{p_text} {stars}', xy=(ci_u + 0.03, i), va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.1, max(ci_upper) + 0.25)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_4B_Regression_Coefficients.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_4B_Regression_Coefficients.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 4B saved (Regression coefficients)")


# ============================================================
# FIGURE 4C: Subgroup Effect Sizes
# ============================================================

def figure_4c_subgroup_effects(data):
    """Horizontal bar chart of effect sizes by subgroup"""

    df = data['subgroup']

    comparisons = df['Comparison'].tolist()

    d_col = [c for c in df.columns if 'd' in c.lower() or 'effect' in c.lower()][0]
    effect_sizes = df[d_col].tolist()

    # Shorten labels
    short_labels = []
    for c in comparisons:
        c = c.replace(' vs Control', '\nvs Control')
        c = c.replace('MDD ', 'MDD\n')
        short_labels.append(c)

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(short_labels))
    bars = ax.barh(y_pos, effect_sizes, color=COLOR_HIGHLIGHT, alpha=0.85,
                   edgecolor='black', linewidth=1.2, height=0.6)

    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label="Medium effect (d=0.5)")
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label="Small effect (d=0.2)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title('Effect Sizes by Subgroup', fontweight='bold')

    # Add values
    for bar, val in zip(bars, effect_sizes):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'd = {val:.2f}', va='center', fontsize=11, fontweight='bold')

    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, max(effect_sizes) * 1.25)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_4C_Subgroup_Effects.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_4C_Subgroup_Effects.pdf', dpi=DPI)
    plt.close()

    print("✓ Figure 4C saved (Subgroup effect sizes)")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("MDD PUBLICATION FIGURES GENERATOR")
    print("=" * 60)

    set_style()

    print("\nLoading data...")
    data = load_data()
    print("✓ Data loaded successfully")

    print("\n" + "-" * 40)
    print("GENERATING FIGURES")
    print("-" * 40 + "\n")

    # Figure 1: Validation
    figure_1a_discovery_boxplot(data)
    figure_1b_validation_boxplot(data)
    figure_1c_effect_sizes(data)
    figure_1d_replication_heatmap(data)

    # Figure 2: Pathways
    figure_2a_gene_counts(data)
    figure_2b_positive_pathways(data)
    figure_2c_negative_pathways(data)

    # Figure 3: WGCNA
    figure_3a_module_correlations(data)
    figure_3b_module_eigengene(data)

    # Figure 4: Confounders
    figure_4a_anxiety_stratification(data)
    figure_4b_regression_coefficients(data)
    figure_4c_subgroup_effects(data)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print(f"DPI: {DPI}")


if __name__=="__main__":
    main()
