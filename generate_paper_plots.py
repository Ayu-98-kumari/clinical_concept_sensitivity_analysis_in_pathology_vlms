"""
Generate comprehensive plots for the paper based on zero-shot evaluation results.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Model names and colors
MODELS = ['CONCH', 'PathGen-CLIP', 'KEEP', 'QuiltNet', 'PLIP']
MODEL_COLORS = {
    'CONCH': '#1f77b4',
    'PathGen-CLIP': '#ff7f0e',
    'KEEP': '#2ca02c',
    'QuiltNet': '#d62728',
    'PLIP': '#9467bd'
}

# Prompt categories
CATEGORIES = ['minimal', 'anatomical', 'descriptive', 'clinical']
CATEGORY_COLORS = {
    'minimal': '#e74c3c',
    'anatomical': '#3498db',
    'descriptive': '#2ecc71',
    'clinical': '#9b59b6'
}

def parse_summary_file(filepath):
    """Parse a summary.txt file and extract all metrics."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract model name
    model_match = re.search(r'Name: (.+)', content)
    model_name = model_match.group(1) if model_match else 'Unknown'

    # Extract per-prompt results
    results = {}
    lines = content.split('\n')
    in_results = False

    for line in lines:
        if 'Per-Prompt Results:' in line:
            in_results = True
            continue
        if in_results and line.strip() and not line.startswith('-') and not line.startswith('Prompt'):
            parts = line.split()
            if len(parts) >= 4:
                prompt = parts[0]
                accuracy = float(parts[1])
                f1 = float(parts[2])
                auc_roc = float(parts[3])

                results[prompt] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'auc_roc': auc_roc
                }

    return model_name, results

def load_all_results():
    """Load results from all summary files."""
    results_dir = Path(__file__).resolve().parent / 'results' / 'zero_shot'
    all_results = {}

    for summary_file in results_dir.glob('*_summary.txt'):
        model_name, results = parse_summary_file(summary_file)
        all_results[model_name] = results

    return all_results

def categorize_prompt(prompt_name):
    """Get category from prompt name."""
    for cat in CATEGORIES:
        if prompt_name.startswith(cat):
            return cat
    return 'unknown'

def create_dataframe(all_results):
    """Create a pandas DataFrame from results."""
    rows = []
    for model, prompts in all_results.items():
        for prompt, metrics in prompts.items():
            row = {
                'model': model,
                'prompt': prompt,
                'category': categorize_prompt(prompt),
                'accuracy': metrics['accuracy'] * 100,  # Convert to percentage
                'f1': metrics['f1'] * 100,
                'auc_roc': metrics['auc_roc'] * 100
            }
            rows.append(row)

    return pd.DataFrame(rows)

def plot_category_comparison(df, output_dir):
    """Plot 1: Mean performance by category for each model with std error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy (%)', 'F1-Score (%)', 'AUC-ROC (%)']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Calculate mean and std by category for each model
        category_means = df.groupby(['model', 'category'])[metric].mean().unstack()
        category_stds = df.groupby(['model', 'category'])[metric].std().unstack()
        category_means = category_means[CATEGORIES]  # Ensure consistent order
        category_stds = category_stds[CATEGORIES]

        # Plot grouped bar chart with error bars
        x = np.arange(len(CATEGORIES))
        width = 0.15

        for i, model in enumerate(MODELS):
            if model in category_means.index:
                values = category_means.loc[model].values
                errors = category_stds.loc[model].values
                ax.bar(x + i*width, values, width, label=model,
                       color=MODEL_COLORS[model], alpha=0.8,
                       yerr=errors, capsize=3, error_kw={'linewidth': 1, 'elinewidth': 1})

        ax.set_xlabel('Prompt Category')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Prompt Category (Mean ± SD)')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([c.capitalize() for c in CATEGORIES], rotation=0)
        ax.legend(loc='best', ncol=2, fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])  # Slightly higher to accommodate error bars

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_category_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_category_comparison.png'}")
    plt.close()

def plot_performance_variation(df, output_dir):
    """Plot 2: Box plot showing variation within categories."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy (%)', 'F1-Score (%)', 'AUC-ROC (%)']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Prepare data for box plot
        data_by_category = [df[df['category'] == cat][metric].values
                           for cat in CATEGORIES]

        bp = ax.boxplot(data_by_category, labels=[c.capitalize() for c in CATEGORIES],
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

        # Color boxes
        for patch, cat in zip(bp['boxes'], CATEGORIES):
            patch.set_facecolor(CATEGORY_COLORS[cat])
            patch.set_alpha(0.6)

        ax.set_xlabel('Prompt Category')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Distribution by Category')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_performance_variation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_performance_variation.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_performance_variation.png'}")
    plt.close()

def plot_model_heatmaps(df, output_dir):
    """Plot 3: Heatmaps for each metric showing model × category performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy (%)', 'F1-Score (%)', 'AUC-ROC (%)']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Create pivot table: models × categories
        pivot = df.groupby(['model', 'category'])[metric].mean().unstack()
        pivot = pivot.reindex(MODELS)[CATEGORIES]  # Consistent ordering

        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   vmin=40, vmax=90, cbar_kws={'label': metric_name},
                   ax=ax, linewidths=0.5)

        ax.set_title(f'{metric_name}: Model × Category')
        ax.set_xlabel('Prompt Category')
        ax.set_ylabel('Model')
        ax.set_xticklabels([c.capitalize() for c in CATEGORIES], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_model_category_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_model_category_heatmaps.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_model_category_heatmaps.png'}")
    plt.close()

def plot_prompt_sensitivity(df, output_dir):
    """Plot 4: Model-specific prompt sensitivity (max - min performance)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy', 'F1-Score', 'AUC-ROC']

    x = np.arange(len(MODELS))
    width = 0.25

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ranges = []
        for model in MODELS:
            model_data = df[df['model'] == model][metric]
            range_val = model_data.max() - model_data.min()
            ranges.append(range_val)

        ax.bar(x + idx*width, ranges, width, label=metric_name, alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Performance Range (Max - Min) (%)')
    ax.set_title('Model Sensitivity to Prompt Choice')
    ax.set_xticks(x + width)
    ax.set_xticklabels(MODELS, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_prompt_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_prompt_sensitivity.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_prompt_sensitivity.png'}")
    plt.close()

def plot_best_worst_comparison(df, output_dir):
    """Plot 5: Best vs Worst performance per model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy (%)', 'F1-Score (%)', 'AUC-ROC (%)']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        best_vals = []
        worst_vals = []

        for model in MODELS:
            model_data = df[df['model'] == model][metric]
            best_vals.append(model_data.max())
            worst_vals.append(model_data.min())

        x = np.arange(len(MODELS))
        width = 0.35

        ax.bar(x - width/2, best_vals, width, label='Best Prompt',
               color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, worst_vals, width, label='Worst Prompt',
               color='#e74c3c', alpha=0.8)

        # Add improvement annotation
        for i, (best, worst) in enumerate(zip(best_vals, worst_vals)):
            improvement = best - worst
            ax.text(i, best + 1, f'+{improvement:.1f}%',
                   ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}: Best vs Worst Prompt')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_best_worst_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_best_worst_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_best_worst_comparison.png'}")
    plt.close()

def plot_improvement_analysis(df, output_dir):
    """Plot 6: Improvement from minimal to clinical/descriptive categories."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ['accuracy', 'f1', 'auc_roc']
    metric_names = ['Accuracy', 'F1-Score', 'AUC-ROC']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        improvements_clinical = []
        improvements_descriptive = []

        for model in MODELS:
            minimal_mean = df[(df['model'] == model) & (df['category'] == 'minimal')][metric].mean()
            clinical_mean = df[(df['model'] == model) & (df['category'] == 'clinical')][metric].mean()
            descriptive_mean = df[(df['model'] == model) & (df['category'] == 'descriptive')][metric].mean()

            improvements_clinical.append(clinical_mean - minimal_mean)
            improvements_descriptive.append(descriptive_mean - minimal_mean)

        x = np.arange(len(MODELS))
        width = 0.35

        ax.bar(x - width/2, improvements_clinical, width,
               label='Clinical vs Minimal', color='#9b59b6', alpha=0.8)
        ax.bar(x + width/2, improvements_descriptive, width,
               label='Descriptive vs Minimal', color='#2ecc71', alpha=0.8)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel(f'{metric_name} Improvement (%)')
        ax.set_title(f'{metric_name} Improvement over Minimal Prompts')
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_improvement_analysis.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_improvement_analysis.png'}")
    plt.close()

def create_extended_results_table(df, output_dir):
    """Create LaTeX table with all metrics (Accuracy, F1, AUC-ROC)."""
    # Get best prompt per model per metric
    best_results = {}

    for model in MODELS:
        model_df = df[df['model'] == model]
        best_results[model] = {
            'accuracy': {
                'value': model_df['accuracy'].max(),
                'prompt': model_df.loc[model_df['accuracy'].idxmax(), 'prompt']
            },
            'f1': {
                'value': model_df['f1'].max(),
                'prompt': model_df.loc[model_df['f1'].idxmax(), 'prompt']
            },
            'auc_roc': {
                'value': model_df['auc_roc'].max(),
                'prompt': model_df.loc[model_df['auc_roc'].idxmax(), 'prompt']
            }
        }

    # Create LaTeX table
    latex = r"""\begin{table}[t]
\centering
\caption{Comprehensive Performance Metrics for All Models}
\label{tab:all_metrics}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}lcccccc@{}}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Accuracy (\%)}} & \multicolumn{2}{c}{\textbf{F1-Score (\%)}} & \multicolumn{2}{c}{\textbf{AUC-ROC (\%)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{Best} & \textbf{Prompt} & \textbf{Best} & \textbf{Prompt} & \textbf{Best} & \textbf{Prompt} \\ \midrule
"""

    for model in MODELS:
        acc_val = best_results[model]['accuracy']['value']
        acc_prompt = best_results[model]['accuracy']['prompt']
        f1_val = best_results[model]['f1']['value']
        f1_prompt = best_results[model]['f1']['prompt']
        auc_val = best_results[model]['auc_roc']['value']
        auc_prompt = best_results[model]['auc_roc']['prompt']

        # Highlight if same prompt achieves best across metrics
        latex += f"{model} & {acc_val:.1f} & {acc_prompt.replace('_', ' ')} & "
        latex += f"{f1_val:.1f} & {f1_prompt.replace('_', ' ')} & "
        latex += f"{auc_val:.1f} & {auc_prompt.replace('_', ' ')} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

    with open(output_dir / 'table_all_metrics.tex', 'w') as f:
        f.write(latex)

    print(f"Saved: {output_dir / 'table_all_metrics.tex'}")

def main():
    """Generate all plots."""
    print("="*80)
    print("GENERATING PAPER PLOTS")
    print("="*80)

    # Load all results
    print("\n[1/8] Loading results...")
    all_results = load_all_results()
    df = create_dataframe(all_results)
    print(f"Loaded {len(df)} data points from {len(all_results)} models")

    # Create output directory
    output_dir = Path(__file__).resolve().parent / 'paper_figures'
    output_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\n[2/8] Generating category comparison plot...")
    plot_category_comparison(df, output_dir)

    print("\n[3/8] Generating performance variation plot...")
    plot_performance_variation(df, output_dir)

    print("\n[4/8] Generating model-category heatmaps...")
    plot_model_heatmaps(df, output_dir)

    print("\n[5/8] Generating prompt sensitivity analysis...")
    plot_prompt_sensitivity(df, output_dir)

    print("\n[6/8] Generating best vs worst comparison...")
    plot_best_worst_comparison(df, output_dir)

    print("\n[7/8] Generating improvement analysis...")
    plot_improvement_analysis(df, output_dir)

    print("\n[8/8] Creating extended metrics table...")
    create_extended_results_table(df, output_dir)

    # Print summary statistics for the "30% improvement" claim
    print("\n" + "="*80)
    print("VERIFICATION OF KEY CLAIMS")
    print("="*80)

    print("\n30% Improvement Claim Analysis:")
    for model in MODELS:
        model_df = df[df['model'] == model]
        minimal_mean = model_df[model_df['category'] == 'minimal']['accuracy'].mean()
        best_overall = model_df['accuracy'].max()
        improvement = best_overall - minimal_mean
        improvement_pct = (improvement / minimal_mean) * 100 if minimal_mean > 0 else 0

        print(f"  {model:15s}: {minimal_mean:5.1f}% (minimal) → {best_overall:5.1f}% (best) "
              f"= +{improvement:5.1f}% (+{improvement_pct:4.1f}%)")

    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
