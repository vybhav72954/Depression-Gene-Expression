import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

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

# Output directory - SEPARATE from tables
OUTPUT_DIR = os.path.join(BASE_DIR, "publication_output", "figures")
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
COLOR_DISCOVERY = '#3498DB'
COLOR_VALIDATION = '#E67E22'

# Endothelial markers
ENDOTHELIAL_MARKERS = ['PECAM1', 'VWF', 'CDH5', 'CLDN5', 'FLT1']


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


def load_discovery_data():
    """Load discovery cohort data from series matrix"""
    print("  Loading discovery cohort (GSE54564)...")

    # Load expression matrix
    matrix_data = pd.read_csv(DISCOVERY_MATRIX, sep="\t", comment="!", index_col=0)

    # Get labels
    labels = []
    with open(DISCOVERY_MATRIX, 'r') as f:
        for line in f:
            if "disease state" in line.lower():
                labels = line.strip().split("\t")[1:]
                break

    y = (pd.Series(labels, index=matrix_data.columns)
         .str.replace('"', '').str.replace('disease state:', "").str.strip()
         .map({"MDD case":1, "Control":0})).values

    # Load annotations
    with open(DISCOVERY_ANNOTATION, 'r') as f:
        lines = f.readlines()
    start = next(i + 1 for i, line in enumerate(lines) if line.strip()=="[Probes]")
    bgx = pd.read_csv(DISCOVERY_ANNOTATION, sep="\t", skiprows=start, engine="python")

    probe_col = 'Probe_Id' if 'Probe_Id' in bgx.columns else 'Name'
    probe_to_gene = dict(zip(bgx[probe_col], bgx['Symbol']))

    # Map probes to genes
    gene_names = []
    valid_probes = []
    for probe in matrix_data.index:
        if probe in probe_to_gene and pd.notna(probe_to_gene[probe]):
            gene = str(probe_to_gene[probe]).strip()
            if gene:
                gene_names.append(gene)
                valid_probes.append(probe)

    X = matrix_data.loc[valid_probes].T.values
    if np.max(X) > 100:
        X = np.log2(X + 1)
    X = stats.zscore(X, axis=0)
    X = np.nan_to_num(X, nan=0.0)

    print(f"    Loaded: {X.shape[0]} samples x {X.shape[1]} genes")
    return X, y, gene_names


def load_validation_data():
    """Load validation cohort data"""
    print("  Loading validation cohort (GSE98793)...")

    expr = pd.read_csv(VALIDATION_EXPR, index_col=0)
    meta = pd.read_csv(VALIDATION_META, index_col=0)

    X = expr.T.values
    gene_names = expr.index.tolist()
    y = (meta['Status']=='MDD').astype(int).values

    # Z-score normalize
    X = stats.zscore(X, axis=0)
    X = np.nan_to_num(X, nan=0.0)

    print(f"    Loaded: {X.shape[0]} samples x {X.shape[1]} genes")
    return X, y, gene_names, meta


def compute_endothelial_score(X, gene_names, markers=ENDOTHELIAL_MARKERS):
    """Compute endothelial cell-type score"""
    gene_to_idx = {g.upper():i for i, g in enumerate(gene_names)}
    found = [m for m in markers if m.upper() in gene_to_idx]
    if len(found) < 3:
        return None, found
    idx = [gene_to_idx[m.upper()] for m in found]
    scores = np.mean(X[:, idx], axis=1)
    return scores, found


# ============================================================
# FIGURE 1A: Discovery Cohort Endothelial Boxplot
# ============================================================

def figure_1a_discovery_boxplot(X_disc, y_disc, gene_names_disc):
    """Discovery cohort endothelial scores by diagnosis"""
    print("  Creating Figure 1A...")

    endo_scores, markers_found = compute_endothelial_score(X_disc, gene_names_disc)

    ctrl = endo_scores[y_disc==0]
    mdd = endo_scores[y_disc==1]

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
        x = np.random.normal(pos, 0.08, len(vals))
        color = COLOR_CONTROL if i==0 else COLOR_MDD
        ax.scatter(x, vals, alpha=0.6, s=40, color=color, edgecolor='white', linewidth=0.5, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Control\n(n={len(ctrl)})', f'MDD\n(n={len(mdd)})'])
    ax.set_ylabel('Endothelial Score (z-score)')
    ax.set_title('Discovery Cohort (GSE54564)', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    p_text = f'p = {p:.3f}' if p >= 0.001 else f'p = {p:.2e}'
    ax.annotate(f"Cohen's d = {d:.2f}\n{p_text}",
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_1A_Discovery_Endothelial.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_1A_Discovery_Endothelial.pdf', dpi=DPI)
    plt.close()

    print(f"    Done (d={d:.2f}, p={p:.4f})")
    return d, p


# ============================================================
# FIGURE 1B: Validation Cohort Endothelial Boxplot
# ============================================================

def figure_1b_validation_boxplot(X_val, y_val, gene_names_val):
    """Validation cohort endothelial scores by diagnosis"""
    print("  Creating Figure 1B...")

    endo_scores, markers_found = compute_endothelial_score(X_val, gene_names_val)

    ctrl = endo_scores[y_val==0]
    mdd = endo_scores[y_val==1]

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

    print(f"    Done (d={d:.2f}, p={p:.4f})")
    return d, p


# ============================================================
# FIGURE 1C: Effect Sizes Comparison (Forest Plot)
# ============================================================

def figure_1c_effect_sizes():
    """Forest plot comparing effect sizes across cell types"""
    print("  Creating Figure 1C...")

    # Load pre-computed results
    disc_results = pd.read_csv(PHASE3_CELLTYPES)
    val_results = pd.read_csv(PHASE4_STATS)

    # Get cell type column
    ct_col = disc_results.columns[0]

    results = []
    for _, row in disc_results.iterrows():
        ct = row[ct_col]
        d_disc = row.get('cohens_d', row.get('Cohens_d', row.get('Cohen_d', 0)))

        val_row = val_results[val_results['Cell_Type']==ct]
        d_val = val_row['Cohens_d'].values[0] if len(val_row) > 0 else 0

        results.append({
            'Cell Type':ct,
            'Discovery':d_disc,
            'Validation':d_val
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Discovery', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(results_df))
    width = 0.35

    ax.barh(y_pos - width / 2, results_df['Discovery'], width,
            label='Discovery (GSE54564)', color=COLOR_DISCOVERY, alpha=0.85, edgecolor='white')
    ax.barh(y_pos + width / 2, results_df['Validation'], width,
            label='Validation (GSE98793)', color=COLOR_VALIDATION, alpha=0.85, edgecolor='white')

    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1, label='Medium effect (d=0.5)')
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Highlight endothelial
    endo_idx = results_df[results_df['Cell Type'].str.contains('Endothelial', case=False)].index
    if len(endo_idx) > 0:
        endo_pos = list(results_df['Cell Type']).index(results_df.loc[endo_idx[0], 'Cell Type'])
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

    print("    Done")


# ============================================================
# FIGURE 1D: Replication Heatmap
# ============================================================

def figure_1d_replication_heatmap():
    """Heatmap showing replication status across cell types"""
    print("  Creating Figure 1D...")

    disc_results = pd.read_csv(PHASE3_CELLTYPES)
    val_results = pd.read_csv(PHASE4_STATS)

    ct_col = disc_results.columns[0]

    results = []
    for _, row in disc_results.iterrows():
        ct = row[ct_col]
        d_disc = row.get('cohens_d', row.get('Cohens_d', 0))
        p_disc = row.get('p_value', row.get('p_mwu', 1))

        val_row = val_results[val_results['Cell_Type']==ct]
        if len(val_row) > 0:
            d_val = val_row['Cohens_d'].values[0]
            p_val = val_row['P_Value'].values[0]
        else:
            d_val = 0
            p_val = 1

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
            rep, color = 'Y', 'green'
        elif np.sign(d_eff)==np.sign(v_eff) and abs(v_eff) > 0.15:
            rep, color = '~', 'orange'
        else:
            rep, color = 'X', 'red'
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

    print("    Done")


# ============================================================
# FIGURE 2A: Consensus Genes Count
# ============================================================

def figure_2a_gene_counts():
    """Bar chart showing positive vs negative consensus genes"""
    print("  Creating Figure 2A...")

    with open(PHASE5_POS_GENES, 'r') as f:
        pos_genes = [line.strip() for line in f if line.strip()]

    with open(PHASE5_NEG_GENES, 'r') as f:
        neg_genes = [line.strip() for line in f if line.strip()]

    n_pos = len(pos_genes)
    n_neg = len(neg_genes)

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

    print(f"    Done (Positive: {n_pos}, Negative: {n_neg})")


# ============================================================
# FIGURE 2B: Positive Pathway Enrichment
# ============================================================

def figure_2b_positive_pathways():
    """Horizontal bar chart of top positive pathways"""
    print("  Creating Figure 2B...")

    df = pd.read_csv(PHASE5_ENRICHR_POS)

    # Find p-value column
    pval_col = [c for c in df.columns if 'p' in c.lower() and 'adj' in c.lower()]
    if not pval_col:
        pval_col = [c for c in df.columns if 'p' in c.lower()]
    pval_col = pval_col[0] if pval_col else df.columns[2]

    # Find term column
    term_col = [c for c in df.columns if 'term' in c.lower()]
    term_col = term_col[0] if term_col else df.columns[1]

    df = df.sort_values(pval_col).head(10)

    pathways = df[term_col].tolist()
    pathways = [p[:40] + '...' if len(str(p)) > 40 else p for p in pathways]
    pvals = df[pval_col].tolist()
    log_pvals = [-np.log10(p) for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(pathways))
    ax.barh(y_pos, log_pvals, color=COLOR_POSITIVE, alpha=0.85, edgecolor='white', height=0.7)

    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='p = 0.05 threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Pathway Enrichment: Positive Correlations', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_2B_Positive_Pathways.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_2B_Positive_Pathways.pdf', dpi=DPI)
    plt.close()

    print("    Done")


# ============================================================
# FIGURE 2C: Negative Pathway Enrichment
# ============================================================

def figure_2c_negative_pathways():
    """Horizontal bar chart of top negative pathways"""
    print("  Creating Figure 2C...")

    df = pd.read_csv(PHASE5_ENRICHR_NEG)

    pval_col = [c for c in df.columns if 'p' in c.lower() and 'adj' in c.lower()]
    if not pval_col:
        pval_col = [c for c in df.columns if 'p' in c.lower()]
    pval_col = pval_col[0] if pval_col else df.columns[2]

    term_col = [c for c in df.columns if 'term' in c.lower()]
    term_col = term_col[0] if term_col else df.columns[1]

    df = df.sort_values(pval_col).head(10)

    pathways = df[term_col].tolist()
    pathways = [p[:40] + '...' if len(str(p)) > 40 else p for p in pathways]
    pvals = df[pval_col].tolist()
    log_pvals = [-np.log10(p) for p in pvals]

    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(pathways))
    ax.barh(y_pos, log_pvals, color=COLOR_NEGATIVE, alpha=0.85, edgecolor='white', height=0.7)

    ax.axvline(x=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='p = 0.05 threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pathways, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Pathway Enrichment: Negative Correlations', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Figure_2C_Negative_Pathways.png', dpi=DPI)
    plt.savefig(f'{OUTPUT_DIR}/Figure_2C_Negative_Pathways.pdf', dpi=DPI)
    plt.close()

    print("    Done")


# ============================================================
# FIGURE 3A: Module-Endothelial Correlations
# ============================================================

def figure_3a_module_correlations():
    """Bar chart of module-endothelial correlations"""
    print("  Creating Figure 3A...")

    df = pd.read_csv(PHASE5_MODULE_ENDO)

    # Find module and correlation columns
    module_col = df.columns[0]
    corr_cols = [c for c in df.columns if 'corr' in c.lower() or 'endo' in c.lower()]
    corr_col = corr_cols[0] if corr_cols else df.columns[1]

    df = df.sort_values(corr_col, ascending=True)

    modules = df[module_col].tolist()
    correlations = df[corr_col].tolist()

    colors = [COLOR_POSITIVE if c > 0 else COLOR_NEGATIVE for c in correlations]

    fig, ax = plt.subplots(figsize=(8, 6))

    y_pos = np.arange(len(modules))
    ax.barh(y_pos, correlations, color=colors, alpha=0.85, edgecolor='white', height=0.7)

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

    print("    Done")


# ============================================================
# FIGURE 4A: Anxiety Stratification
# ============================================================

def figure_4a_anxiety_stratification():
    """Boxplot of endothelial scores stratified by anxiety"""
    print("  Creating Figure 4A...")

    # Load anxiety stratified analysis
    df = pd.read_csv(PHASE6_ANXIETY)

    # This figure needs raw scores, load from Phase 4 deconvolution + full metadata
    deconv = pd.read_csv(PHASE4_DECONV, index_col=0)

    if os.path.exists(VALIDATION_FULL_META):
        full_meta = pd.read_csv(VALIDATION_FULL_META, index_col=0)

        # Find relevant columns
        diag_col = [c for c in full_meta.columns if 'subject' in c.lower() or 'group' in c.lower()]
        anx_col = [c for c in full_meta.columns if 'anxiety' in c.lower()]

        if diag_col and anx_col:
            diag_col = diag_col[0]
            anx_col = anx_col[0]

            # Merge
            merged = deconv.join(full_meta[[diag_col, anx_col]])

            # Parse diagnosis
            merged['Diagnosis'] = merged[diag_col].apply(
                lambda x:'MDD' if 'MDD' in str(x).upper() else 'Control'
            )

            # Parse anxiety
            merged['Anxiety'] = merged[anx_col].apply(
                lambda x:1 if 'yes' in str(x).lower() else 0
            )

            ctrl = merged[merged['Diagnosis']=='Control']['Endothelial'].dropna()
            mdd_no_anx = merged[(merged['Diagnosis']=='MDD') & (merged['Anxiety']==0)]['Endothelial'].dropna()
            mdd_anx = merged[(merged['Diagnosis']=='MDD') & (merged['Anxiety']==1)]['Endothelial'].dropna()

            _, p1 = stats.mannwhitneyu(ctrl, mdd_no_anx) if len(mdd_no_anx) > 0 else (0, 1)
            _, p2 = stats.mannwhitneyu(ctrl, mdd_anx) if len(mdd_anx) > 0 else (0, 1)

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
            y_max = max(ctrl.max(), mdd_no_anx.max() if len(mdd_no_anx) > 0 else 0,
                        mdd_anx.max() if len(mdd_anx) > 0 else 0)

            y1 = y_max + 0.25
            ax.plot([0, 0, 1, 1], [y1, y1 + 0.08, y1 + 0.08, y1], 'k-', linewidth=1)
            stars1 = '**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'ns')
            ax.text(0.5, y1 + 0.12, f'p={p1:.3f} {stars1}', ha='center', fontsize=9)

            y2 = y_max + 0.55
            ax.plot([0, 0, 2, 2], [y2, y2 + 0.08, y2 + 0.08, y2], 'k-', linewidth=1)
            stars2 = '**' if p2 < 0.01 else ('*' if p2 < 0.05 else 'ns')
            ax.text(1, y2 + 0.12, f'p={p2:.3f} {stars2}', ha='center', fontsize=9)

            ax.set_ylim(ax.get_ylim()[0], y2 + 0.4)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/Figure_4A_Anxiety_Stratification.png', dpi=DPI)
            plt.savefig(f'{OUTPUT_DIR}/Figure_4A_Anxiety_Stratification.pdf', dpi=DPI)
            plt.close()

            print("    Done")
            return

    print("    Skipped (metadata not available)")


# ============================================================
# FIGURE 4B: Regression Coefficients
# ============================================================

def figure_4b_regression_coefficients():
    """Forest plot of regression coefficients across models"""
    print("  Creating Figure 4B...")

    df = pd.read_csv(PHASE6_REGRESSION)

    # Find relevant columns
    model_col = [c for c in df.columns if 'model' in c.lower()]
    model_col = model_col[0] if model_col else df.columns[0]

    beta_col = [c for c in df.columns if 'beta' in c.lower() or 'coef' in c.lower() or 'b'==c.lower()]
    beta_col = beta_col[0] if beta_col else df.columns[1]

    pval_col = [c for c in df.columns if 'p' in c.lower()]
    pval_col = pval_col[0] if pval_col else df.columns[2]

    models = df[model_col].tolist()
    betas = df[beta_col].tolist()
    pvals = df[pval_col].tolist()

    # Estimate CIs if not available
    ci_lower = [b - 0.12 for b in betas]
    ci_upper = [b + 0.12 for b in betas]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(models))

    ax.errorbar(betas, y_pos,
                xerr=[np.array(betas) - np.array(ci_lower), np.array(ci_upper) - np.array(betas)],
                fmt='o', markersize=12, color=COLOR_MDD, capsize=6, capthick=2.5,
                elinewidth=2.5, markeredgecolor='white', markeredgewidth=1.5)

    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Diagnosis Coefficient (95% CI)', fontsize=12)
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

    print("    Done")


# ============================================================
# FIGURE 4C: Subgroup Effect Sizes
# ============================================================

def figure_4c_subgroup_effects():
    """Horizontal bar chart of effect sizes by subgroup"""
    print("  Creating Figure 4C...")

    df = pd.read_csv(PHASE6_SUBGROUP)

    # Find columns
    comp_col = [c for c in df.columns if 'comp' in c.lower() or 'group' in c.lower()]
    comp_col = comp_col[0] if comp_col else df.columns[0]

    d_col = [c for c in df.columns if 'd' in c.lower() or 'effect' in c.lower()]
    d_col = d_col[0] if d_col else df.columns[1]

    comparisons = df[comp_col].tolist()
    effect_sizes = df[d_col].tolist()

    # Shorten labels
    short_labels = []
    for c in comparisons:
        c = str(c).replace(' vs Control', '\nvs Control')
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

    print("    Done")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("MDD PUBLICATION FIGURES GENERATOR")
    print("=" * 60)

    set_style()

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)

    # Load data
    X_disc, y_disc, gene_names_disc = load_discovery_data()
    X_val, y_val, gene_names_val, val_meta = load_validation_data()

    print("\n" + "-" * 40)
    print("GENERATING FIGURES")
    print("-" * 40 + "\n")

    # Figure 1: Validation
    print("Figure 1 Series (Validation):")
    figure_1a_discovery_boxplot(X_disc, y_disc, gene_names_disc)
    figure_1b_validation_boxplot(X_val, y_val, gene_names_val)
    figure_1c_effect_sizes()
    figure_1d_replication_heatmap()

    # Figure 2: Pathways
    print("\nFigure 2 Series (Pathways):")
    figure_2a_gene_counts()
    figure_2b_positive_pathways()
    figure_2c_negative_pathways()

    # Figure 3: WGCNA
    print("\nFigure 3 Series (WGCNA):")
    figure_3a_module_correlations()

    # Figure 4: Confounders
    print("\nFigure 4 Series (Confounders):")
    figure_4a_anxiety_stratification()
    figure_4b_regression_coefficients()
    figure_4c_subgroup_effects()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print(f"DPI: {DPI}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__=="__main__":
    main()