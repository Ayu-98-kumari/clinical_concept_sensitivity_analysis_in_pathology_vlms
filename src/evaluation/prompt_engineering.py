"""
Prompt engineering utilities for zero-shot evaluation.

This module provides utilities for loading, managing, and testing
different prompt variations for vision-language models.
"""

from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path
import itertools


class PromptManager:
    """
    Manager for loading and organizing prompts from configuration.

    Example:
        >>> manager = PromptManager('config/prompt_configs.yaml')
        >>> prompts = manager.get_prompts_by_category('minimal')
        >>> all_prompts = manager.get_all_prompts()
    """

    def __init__(self, config_path: str):
        """
        Initialize prompt manager.

        Args:
            config_path: Path to prompt configuration YAML file
        """
        self.config_path = Path(config_path)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load prompts from YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if 'prompt_categories' not in config:
            raise ValueError("Config must contain 'prompt_categories' key")

        return config['prompt_categories']

    def get_categories(self) -> List[str]:
        """
        Get list of prompt categories.

        Returns:
            List of category names

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> categories = manager.get_categories()
            >>> print(categories)
            ['minimal', 'anatomical', 'descriptive', 'clinical']
        """
        return list(self.prompts.keys())

    def get_prompts_by_category(
        self,
        category: str
    ) -> List[Tuple[str, str]]:
        """
        Get prompts for a specific category.

        Args:
            category: Category name

        Returns:
            List of (positive_prompt, negative_prompt) tuples

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> prompts = manager.get_prompts_by_category('minimal')
            >>> print(prompts[0])
            ('cancer', 'normal')
        """
        if category not in self.prompts:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Available: {list(self.prompts.keys())}"
            )
        return self.prompts[category]

    def get_all_prompts(self) -> List[Tuple[str, str, str]]:
        """
        Get all prompts with their categories.

        Returns:
            List of (category, positive_prompt, negative_prompt) tuples

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> all_prompts = manager.get_all_prompts()
            >>> print(f"Total prompts: {len(all_prompts)}")
            Total prompts: 20
        """
        all_prompts = []
        for category, prompt_list in self.prompts.items():
            for positive, negative in prompt_list:
                all_prompts.append((category, positive, negative))
        return all_prompts

    def get_prompt_by_index(
        self,
        category: str,
        index: int
    ) -> Tuple[str, str]:
        """
        Get a specific prompt by category and index.

        Args:
            category: Category name
            index: Prompt index within category

        Returns:
            Tuple of (positive_prompt, negative_prompt)

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> prompt = manager.get_prompt_by_index('minimal', 0)
            >>> print(prompt)
            ('cancer', 'normal')
        """
        prompts = self.get_prompts_by_category(category)
        if index < 0 or index >= len(prompts):
            raise IndexError(
                f"Index {index} out of range for category {category} "
                f"(has {len(prompts)} prompts)"
            )
        return prompts[index]

    def get_num_prompts(self) -> int:
        """
        Get total number of prompts across all categories.

        Returns:
            Total number of prompts

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> print(manager.get_num_prompts())
            20
        """
        return sum(len(prompts) for prompts in self.prompts.values())

    def get_prompt_statistics(self) -> Dict[str, int]:
        """
        Get statistics about prompts.

        Returns:
            Dictionary with prompt statistics

        Example:
            >>> manager = PromptManager('config/prompt_configs.yaml')
            >>> stats = manager.get_prompt_statistics()
            >>> print(stats)
            {'total': 20, 'minimal': 5, 'anatomical': 5, ...}
        """
        stats = {'total': self.get_num_prompts()}
        for category, prompt_list in self.prompts.items():
            stats[category] = len(prompt_list)
        return stats


def create_prompt_pairs(
    class_names: List[str],
    template: str = "{class_name}"
) -> List[Tuple[str, str]]:
    """
    Create prompt pairs from class names using a template.

    Args:
        class_names: List of class names
        template: Template string with {class_name} placeholder

    Returns:
        List of prompt pairs (for binary classification)

    Example:
        >>> class_names = ['tumor', 'normal']
        >>> prompts = create_prompt_pairs(class_names)
        >>> print(prompts)
        [('tumor', 'normal')]

        >>> prompts = create_prompt_pairs(
        ...     class_names,
        ...     template="histopathology image of {class_name}"
        ... )
        >>> print(prompts[0])
        ('histopathology image of tumor', 'histopathology image of normal')
    """
    if len(class_names) == 2:
        # Binary classification
        return [(
            template.format(class_name=class_names[0]),
            template.format(class_name=class_names[1])
        )]
    else:
        # Multi-class: create all pairs
        pairs = []
        for i, j in itertools.combinations(range(len(class_names)), 2):
            pairs.append((
                template.format(class_name=class_names[i]),
                template.format(class_name=class_names[j])
            ))
        return pairs


def create_descriptive_prompts(
    base_class_names: List[str],
    modality: str = "histopathology",
    staining: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Create descriptive prompts with imaging modality information.

    Args:
        base_class_names: Base class names
        modality: Imaging modality (e.g., 'histopathology', 'microscopy')
        staining: Optional staining method (e.g., 'H&E')

    Returns:
        List of prompt pairs

    Example:
        >>> prompts = create_descriptive_prompts(
        ...     ['cancer', 'normal'],
        ...     modality='histopathology',
        ...     staining='H&E'
        ... )
        >>> print(prompts[0])
        ('H&E stained histopathology image showing cancer',
         'H&E stained histopathology image showing normal')
    """
    prompts = []

    for pos_class, neg_class in zip(base_class_names[::2], base_class_names[1::2]):
        if staining:
            pos_prompt = f"{staining} stained {modality} image showing {pos_class}"
            neg_prompt = f"{staining} stained {modality} image showing {neg_class}"
        else:
            pos_prompt = f"{modality} image showing {pos_class}"
            neg_prompt = f"{modality} image showing {neg_class}"

        prompts.append((pos_prompt, neg_prompt))

    return prompts


def analyze_prompt_sensitivity(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy'
) -> Dict[str, float]:
    """
    Analyze sensitivity to different prompts.

    Args:
        results: Dictionary mapping prompt names to metrics
        metric: Metric to analyze

    Returns:
        Dictionary with sensitivity statistics

    Example:
        >>> results = {
        ...     'prompt1': {'accuracy': 0.85},
        ...     'prompt2': {'accuracy': 0.87},
        ...     'prompt3': {'accuracy': 0.86}
        ... }
        >>> sensitivity = analyze_prompt_sensitivity(results)
        >>> print(sensitivity)
        {'mean': 0.86, 'std': 0.01, 'min': 0.85, 'max': 0.87, ...}
    """
    import numpy as np

    values = [r[metric] for r in results.values() if metric in r]

    if not values:
        return {}

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'coefficient_of_variation': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0
    }


def find_best_prompt(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy'
) -> Tuple[str, float]:
    """
    Find the best performing prompt.

    Args:
        results: Dictionary mapping prompt names to metrics
        metric: Metric to use for comparison

    Returns:
        Tuple of (best_prompt_name, best_metric_value)

    Example:
        >>> results = {
        ...     'prompt1': {'accuracy': 0.85},
        ...     'prompt2': {'accuracy': 0.87},
        ...     'prompt3': {'accuracy': 0.86}
        ... }
        >>> best_name, best_value = find_best_prompt(results)
        >>> print(f"Best: {best_name} with {best_value:.3f}")
        Best: prompt2 with 0.870
    """
    best_prompt = None
    best_value = -float('inf')

    for prompt_name, metrics in results.items():
        if metric in metrics:
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best_prompt = prompt_name

    if best_prompt is None:
        raise ValueError(f"Metric '{metric}' not found in results")

    return best_prompt, float(best_value)


def format_prompt_results(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1', 'auc_roc']
) -> str:
    """
    Format prompt evaluation results as a table.

    Args:
        results: Dictionary mapping prompt names to metrics
        metrics: List of metrics to include

    Returns:
        Formatted table string

    Example:
        >>> results = {
        ...     'minimal_1': {'accuracy': 0.85, 'f1': 0.83},
        ...     'minimal_2': {'accuracy': 0.87, 'f1': 0.85}
        ... }
        >>> print(format_prompt_results(results))
    """
    lines = []

    # Header
    header = f"{'Prompt':<30s}"
    for metric in metrics:
        header += f" {metric:>10s}"
    lines.append(header)
    lines.append("-" * (30 + 11 * len(metrics)))

    # Results
    for prompt_name, prompt_metrics in sorted(results.items()):
        row = f"{prompt_name:<30s}"
        for metric in metrics:
            value = prompt_metrics.get(metric, 0.0)
            row += f" {value:>10.4f}"
        lines.append(row)

    return "\n".join(lines)


class PromptTester:
    """
    Utility for testing multiple prompts systematically.

    Example:
        >>> tester = PromptTester(model, dataset)
        >>> results = tester.test_all_prompts(prompt_manager)
        >>> best = tester.get_best_prompt()
    """

    def __init__(self, model, dataset):
        """
        Initialize prompt tester.

        Args:
            model: Vision-language model
            dataset: Dataset to evaluate on
        """
        self.model = model
        self.dataset = dataset
        self.results = {}

    def test_prompt(
        self,
        prompt_pair: Tuple[str, str],
        prompt_name: str
    ) -> Dict[str, float]:
        """
        Test a single prompt pair.

        Args:
            prompt_pair: (positive_prompt, negative_prompt)
            prompt_name: Name for this prompt

        Returns:
            Dictionary of metrics
        """
        # This would be implemented with actual evaluation logic
        # For now, it's a placeholder
        raise NotImplementedError("Use ZeroShotEvaluator for actual testing")

    def test_all_prompts(
        self,
        prompt_manager: PromptManager
    ) -> Dict[str, Dict[str, float]]:
        """
        Test all prompts from manager.

        Args:
            prompt_manager: PromptManager instance

        Returns:
            Dictionary mapping prompt names to metrics
        """
        raise NotImplementedError("Use ZeroShotEvaluator for actual testing")

    def get_best_prompt(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """Get the best performing prompt."""
        return find_best_prompt(self.results, metric)
