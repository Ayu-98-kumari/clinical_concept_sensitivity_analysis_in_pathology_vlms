"""
Generate comprehensive LaTeX tables for all metrics (F1, AUC-ROC, Balanced Accuracy).
Similar to Table 3 in the paper but for other metrics.
"""

import re
from pathlib import Path

MODELS = ['CONCH', 'PathGen-CLIP', 'KEEP', 'QuiltNet', 'PLIP']
CATEGORIES = ['minimal', 'anatomical', 'descriptive', 'clinical']
PROMPTS = [
    'minimal_1', 'minimal_2', 'minimal_3', 'minimal_4', 'minimal_5',
    'anatomical_1', 'anatomical_2', 'anatomical_3', 'anatomical_4', 'anatomical_5',
    'descriptive_1', 'descriptive_2', 'descriptive_3', 'descriptive_4', 'descriptive_5',
    'clinical_1', 'clinical_2', 'clinical_3', 'clinical_4', 'clinical_5'
]

def parse_summary_file(filepath):
    """Parse a summary.txt file and extract all metrics."""
    with open(filepath, 'r') as f:
        content = f.read()

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
                accuracy = float(parts[1]) * 100
                f1 = float(parts[2]) * 100
                auc_roc = float(parts[3]) * 100

                results[prompt] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'auc_roc': auc_roc
                }

    return results

def load_all_results():
    """Load results from all summary files."""
    results_dir = Path(__file__).resolve().parent / 'results' / 'zero_shot'
    all_results = {}

    model_file_mapping = {
        'CONCH': 'conch_zero_shot_summary.txt',
        'PathGen-CLIP': 'pathgenclip_zero_shot_summary.txt',
        'KEEP': 'keep_zero_shot_summary.txt',
        'QuiltNet': 'quiltnet_zero_shot_summary.txt',
        'PLIP': 'plip_zero_shot_summary.txt'
    }

    for model, filename in model_file_mapping.items():
        filepath = results_dir / filename
        results = parse_summary_file(filepath)
        all_results[model] = results

    return all_results

def create_f1_table(all_results, output_path):
    """Create LaTeX table for F1-scores (similar to Table 3)."""
    latex = r"""\begin{table*}[t]
\centering
\caption{Complete F1-Score Results (\%) for All Models and Prompts}
\label{tab:f1_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Category} & \textbf{Prompt} & \textbf{CONCH} & \textbf{PathGen-CLIP} & \textbf{KEEP} & \textbf{QuiltNet} & \textbf{PLIP} \\ \midrule
"""

    for category in CATEGORIES:
        category_prompts = [p for p in PROMPTS if p.startswith(category)]
        for i, prompt in enumerate(category_prompts):
            if i == 0:
                latex += f"\\multirow{{5}}{{*}}{{{category.capitalize()}}}\n"
            else:
                latex += " "

            latex += f"& {prompt.replace('_', '\\_')} "

            for model in MODELS:
                value = all_results[model].get(prompt, {}).get('f1', 0.0)
                latex += f"& {value:.1f} "

            latex += "\\\\\n"

        if category != CATEGORIES[-1]:
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

def create_aucroc_table(all_results, output_path):
    """Create LaTeX table for AUC-ROC scores."""
    latex = r"""\begin{table*}[t]
\centering
\caption{Complete AUC-ROC Results (\%) for All Models and Prompts}
\label{tab:aucroc_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}llccccc@{}}
\toprule
\textbf{Category} & \textbf{Prompt} & \textbf{CONCH} & \textbf{PathGen-CLIP} & \textbf{KEEP} & \textbf{QuiltNet} & \textbf{PLIP} \\ \midrule
"""

    for category in CATEGORIES:
        category_prompts = [p for p in PROMPTS if p.startswith(category)]
        for i, prompt in enumerate(category_prompts):
            if i == 0:
                latex += f"\\multirow{{5}}{{*}}{{{category.capitalize()}}}\n"
            else:
                latex += " "

            latex += f"& {prompt.replace('_', '\\_')} "

            for model in MODELS:
                value = all_results[model].get(prompt, {}).get('auc_roc', 0.0)
                latex += f"& {value:.1f} "

            latex += "\\\\\n"

        if category != CATEGORIES[-1]:
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

def create_category_stats_table(all_results, output_path):
    """Create enhanced category statistics table with all metrics."""
    import numpy as np

    # Calculate statistics for each metric
    stats = {}
    for metric in ['accuracy', 'f1', 'auc_roc']:
        stats[metric] = {}
        for model in MODELS:
            stats[metric][model] = {}
            for category in CATEGORIES:
                category_prompts = [p for p in PROMPTS if p.startswith(category)]
                values = [all_results[model].get(p, {}).get(metric, 0.0)
                         for p in category_prompts if p in all_results[model]]
                if values:
                    stats[metric][model][category] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }

    # Find best prompt per model per metric
    best_prompts = {}
    for metric in ['accuracy', 'f1', 'auc_roc']:
        best_prompts[metric] = {}
        for model in MODELS:
            best_val = 0
            best_prompt = ''
            for prompt in PROMPTS:
                val = all_results[model].get(prompt, {}).get(metric, 0.0)
                if val > best_val:
                    best_val = val
                    best_prompt = prompt
            best_prompts[metric][model] = {'prompt': best_prompt, 'value': best_val}

    # Create table
    latex = r"""\begin{table*}[t]
\centering
\caption{Comprehensive Performance Statistics by Prompt Category and Metric}
\label{tab:comprehensive_category_stats}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccc@{}}
\toprule
"""

    # Accuracy section
    latex += r"""\multicolumn{6}{c}{\textbf{Accuracy (\%) by Category}} \\
\midrule
\textbf{Category} & \textbf{CONCH} & \textbf{PathGen-CLIP} & \textbf{KEEP} & \textbf{QuiltNet} & \textbf{PLIP} \\ \midrule
"""

    for category in CATEGORIES:
        latex += f"{category.capitalize()} "
        for model in MODELS:
            mean = stats['accuracy'][model][category]['mean']
            std = stats['accuracy'][model][category]['std']
            latex += f"& {mean:.1f} $\\pm$ {std:.1f} "
        latex += "\\\\\n"

    latex += "\\midrule\n"
    latex += "\\textbf{Best Prompt} "
    for model in MODELS:
        prompt = best_prompts['accuracy'][model]['prompt'].replace('_', '\\_')
        latex += f"& {prompt} "
    latex += "\\\\\n"

    latex += "\\textbf{Best Value} "
    for model in MODELS:
        value = best_prompts['accuracy'][model]['value']
        latex += f"& {value:.1f} "
    latex += "\\\\\n"

    # F1 section
    latex += r"""\midrule
\multicolumn{6}{c}{\textbf{F1-Score (\%) by Category}} \\
\midrule
\textbf{Category} & \textbf{CONCH} & \textbf{PathGen-CLIP} & \textbf{KEEP} & \textbf{QuiltNet} & \textbf{PLIP} \\ \midrule
"""

    for category in CATEGORIES:
        latex += f"{category.capitalize()} "
        for model in MODELS:
            mean = stats['f1'][model][category]['mean']
            std = stats['f1'][model][category]['std']
            latex += f"& {mean:.1f} $\\pm$ {std:.1f} "
        latex += "\\\\\n"

    latex += "\\midrule\n"
    latex += "\\textbf{Best Prompt} "
    for model in MODELS:
        prompt = best_prompts['f1'][model]['prompt'].replace('_', '\\_')
        latex += f"& {prompt} "
    latex += "\\\\\n"

    latex += "\\textbf{Best Value} "
    for model in MODELS:
        value = best_prompts['f1'][model]['value']
        latex += f"& {value:.1f} "
    latex += "\\\\\n"

    # AUC-ROC section
    latex += r"""\midrule
\multicolumn{6}{c}{\textbf{AUC-ROC (\%) by Category}} \\
\midrule
\textbf{Category} & \textbf{CONCH} & \textbf{PathGen-CLIP} & \textbf{KEEP} & \textbf{QuiltNet} & \textbf{PLIP} \\ \midrule
"""

    for category in CATEGORIES:
        latex += f"{category.capitalize()} "
        for model in MODELS:
            mean = stats['auc_roc'][model][category]['mean']
            std = stats['auc_roc'][model][category]['std']
            latex += f"& {mean:.1f} $\\pm$ {std:.1f} "
        latex += "\\\\\n"

    latex += "\\midrule\n"
    latex += "\\textbf{Best Prompt} "
    for model in MODELS:
        prompt = best_prompts['auc_roc'][model]['prompt'].replace('_', '\\_')
        latex += f"& {prompt} "
    latex += "\\\\\n"

    latex += "\\textbf{Best Value} "
    for model in MODELS:
        value = best_prompts['auc_roc'][model]['value']
        latex += f"& {value:.1f} "
    latex += "\\\\\n"

    latex += r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {output_path}")

def main():
    """Generate all comprehensive tables."""
    print("="*80)
    print("GENERATING COMPREHENSIVE METRICS TABLES")
    print("="*80)

    # Load all results
    print("\n[1/4] Loading results...")
    all_results = load_all_results()
    print(f"Loaded results for {len(all_results)} models")

    # Create output directory
    output_dir = Path(__file__).resolve().parent / 'paper_figures'
    output_dir.mkdir(exist_ok=True)

    # Generate tables
    print("\n[2/4] Generating F1-Score table...")
    create_f1_table(all_results, output_dir / 'table_f1_results.tex')

    print("\n[3/4] Generating AUC-ROC table...")
    create_aucroc_table(all_results, output_dir / 'table_aucroc_results.tex')

    print("\n[4/4] Generating comprehensive category statistics table...")
    create_category_stats_table(all_results, output_dir / 'table_comprehensive_stats.tex')

    print("\n" + "="*80)
    print(f"All tables saved to: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
