"""
Comprehensive evaluation metrics for classification tasks.

This module provides metrics for binary and multi-class classification,
including statistical significance tests and confidence intervals.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef
)
from scipy import stats
import pandas as pd


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_proba: Predicted probabilities [N, 2] or [N] (optional)

    Returns:
        Dictionary of metrics

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = compute_binary_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.8
    """
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positive'] = int(tp)
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)

    # Specificity and sensitivity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Additional metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # AUC-ROC (if probabilities provided)
    if y_proba is not None:
        if y_proba.ndim == 2:
            # [N, 2] format - take positive class probability
            y_proba = y_proba[:, 1]
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0

    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-class classification.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_proba: Predicted probabilities [N, C] (optional)
        num_classes: Number of classes (optional)

    Returns:
        Dictionary of metrics

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 1, 0, 1])
        >>> metrics = compute_multiclass_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.8
    """
    metrics = {}

    # Basic metrics with weighted averaging
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, (f1, prec, rec) in enumerate(zip(per_class_f1, per_class_precision, per_class_recall)):
        metrics[f'f1_class_{i}'] = float(f1)
        metrics[f'precision_class_{i}'] = float(prec)
        metrics[f'recall_class_{i}'] = float(rec)

    # Cohen's Kappa and MCC
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        metrics['mcc'] = 0.0

    # AUC-ROC (if probabilities provided)
    if y_proba is not None:
        try:
            if num_classes is None:
                num_classes = len(np.unique(y_true))
            metrics['auc_roc_ovr'] = roc_auc_score(
                y_true, y_proba,
                multi_class='ovr',
                average='weighted'
            )
        except (ValueError, IndexError):
            metrics['auc_roc_ovr'] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix [C, C]

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> cm = compute_confusion_matrix(y_true, y_pred)
        >>> print(cm)
        [[2 0]
         [1 2]]
    """
    return confusion_matrix(y_true, y_pred)


def calculate_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate mean and confidence interval using t-distribution.

    Args:
        values: List of metric values
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> values = [0.85, 0.87, 0.86, 0.88, 0.84]
        >>> mean, lower, upper = calculate_confidence_interval(values)
        >>> print(f"Mean: {mean:.3f}, CI: [{lower:.3f}, {upper:.3f}]")
        Mean: 0.860, CI: [0.843, 0.877]
    """
    values = np.array(values)
    mean = np.mean(values)
    std_err = stats.sem(values)

    # t-distribution for small samples
    t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    margin = t_value * std_err

    return mean, mean - margin, mean + margin


def statistical_significance_test(
    results_a: List[float],
    results_b: List[float],
    test: str = 'paired_ttest'
) -> Dict[str, float]:
    """
    Perform statistical significance test between two sets of results.

    Args:
        results_a: Results from method A
        results_b: Results from method B
        test: Test type ('paired_ttest', 'wilcoxon', 'independent_ttest')

    Returns:
        Dictionary with test statistic and p-value

    Example:
        >>> results_a = [0.85, 0.87, 0.86, 0.88, 0.84]
        >>> results_b = [0.83, 0.85, 0.84, 0.86, 0.82]
        >>> test_result = statistical_significance_test(results_a, results_b)
        >>> print(f"p-value: {test_result['p_value']:.4f}")
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if test == 'paired_ttest':
        statistic, p_value = stats.ttest_rel(results_a, results_b)
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(results_a, results_b)
    elif test == 'independent_ttest':
        statistic, p_value = stats.ttest_ind(results_a, results_b)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01
    }


def aggregate_metrics_over_runs(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple runs.

    Args:
        metrics_list: List of metric dictionaries from different runs

    Returns:
        Dictionary with mean, std, min, max for each metric

    Example:
        >>> run1 = {'accuracy': 0.85, 'f1': 0.83}
        >>> run2 = {'accuracy': 0.87, 'f1': 0.85}
        >>> run3 = {'accuracy': 0.86, 'f1': 0.84}
        >>> agg = aggregate_metrics_over_runs([run1, run2, run3])
        >>> print(agg['accuracy']['mean'])
        0.86
    """
    if not metrics_list:
        return {}

    # Get all metric names
    metric_names = set()
    for metrics in metrics_list:
        metric_names.update(metrics.keys())

    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics_list if metric_name in m]

        if values:
            mean, ci_lower, ci_upper = calculate_confidence_interval(values)
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }

    return aggregated


def format_metrics_table(
    metrics: Dict[str, float],
    precision: int = 4
) -> str:
    """
    Format metrics as a readable table.

    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places

    Returns:
        Formatted string table

    Example:
        >>> metrics = {'accuracy': 0.8567, 'f1': 0.8234}
        >>> print(format_metrics_table(metrics))
    """
    lines = ["Metric                    Value"]
    lines.append("-" * 40)

    for metric_name, value in sorted(metrics.items()):
        if isinstance(value, (int, np.integer)):
            lines.append(f"{metric_name:25s} {value:6d}")
        else:
            lines.append(f"{metric_name:25s} {value:6.{precision}f}")

    return "\n".join(lines)


class MetricsCalculator:
    """
    Calculator for computing and tracking metrics.

    Example:
        >>> calculator = MetricsCalculator(task='binary')
        >>> metrics = calculator.compute(y_true, y_pred, y_proba)
        >>> calculator.add_run(metrics)
        >>> summary = calculator.get_summary()
    """

    def __init__(self, task: str = 'binary', num_classes: Optional[int] = None):
        """
        Initialize metrics calculator.

        Args:
            task: Task type ('binary' or 'multiclass')
            num_classes: Number of classes (for multiclass)
        """
        self.task = task
        self.num_classes = num_classes
        self.runs = []

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute metrics for given predictions."""
        if self.task == 'binary':
            return compute_binary_metrics(y_true, y_pred, y_proba)
        elif self.task == 'multiclass':
            return compute_multiclass_metrics(y_true, y_pred, y_proba, self.num_classes)
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def add_run(self, metrics: Dict[str, float]):
        """Add metrics from a single run."""
        self.runs.append(metrics)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated summary across all runs."""
        return aggregate_metrics_over_runs(self.runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        if not self.runs:
            return pd.DataFrame()
        return pd.DataFrame(self.runs)

    def compare_with(
        self,
        other: 'MetricsCalculator',
        metric_name: str = 'accuracy'
    ) -> Dict:
        """Compare metrics with another calculator."""
        values_a = [run[metric_name] for run in self.runs if metric_name in run]
        values_b = [run[metric_name] for run in other.runs if metric_name in run]

        if not values_a or not values_b:
            return {'error': 'Insufficient data for comparison'}

        return statistical_significance_test(values_a, values_b)
