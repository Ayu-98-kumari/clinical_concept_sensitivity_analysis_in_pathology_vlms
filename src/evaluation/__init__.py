"""
Evaluation module exports.
"""

from .metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_confusion_matrix,
    calculate_confidence_interval,
    statistical_significance_test,
    aggregate_metrics_over_runs,
    MetricsCalculator
)
from .prompt_engineering import (
    PromptManager,
    create_prompt_pairs,
    analyze_prompt_sensitivity,
    find_best_prompt
)
from .zero_shot import ZeroShotEvaluator

__all__ = [
    # Metrics
    'compute_binary_metrics',
    'compute_multiclass_metrics',
    'compute_confusion_matrix',
    'calculate_confidence_interval',
    'statistical_significance_test',
    'aggregate_metrics_over_runs',
    'MetricsCalculator',
    # Prompt engineering
    'PromptManager',
    'create_prompt_pairs',
    'analyze_prompt_sensitivity',
    'find_best_prompt',
    # Zero-shot evaluation
    'ZeroShotEvaluator'
]
