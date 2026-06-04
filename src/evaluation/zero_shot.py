"""
Zero-shot evaluation for vision-language models.

This module provides functionality for evaluating VLMs in a zero-shot setting
using different text prompts without any training.
"""

from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path

from .metrics import MetricsCalculator, aggregate_metrics_over_runs
from .prompt_engineering import PromptManager, analyze_prompt_sensitivity, find_best_prompt


class ZeroShotEvaluator:
    """
    Evaluator for zero-shot classification using vision-language models.

    Example:
        >>> from src.models import create_model
        >>> from src.data import PCamDataset
        >>>
        >>> model = create_model('conch', config_path='config/model_configs.yaml')
        >>> dataset = PCamDataset(data_dir='path/to/pcam', split='test')
        >>> evaluator = ZeroShotEvaluator(model, batch_size=64)
        >>>
        >>> prompt_manager = PromptManager('config/prompt_configs.yaml')
        >>> results = evaluator.evaluate_all_prompts(dataset, prompt_manager)
    """

    def __init__(
        self,
        model,
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[str] = None
    ):
        """
        Initialize zero-shot evaluator.

        Args:
            model: Vision-language model (must inherit from BaseVLModel)
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            device: Device to use (None = auto-detect)
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def evaluate(
        self,
        dataset,
        prompt_pair: Tuple[str, str],
        prompt_name: str = "default"
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset with a specific prompt pair.

        Args:
            dataset: Dataset to evaluate on
            prompt_pair: Tuple of (positive_prompt, negative_prompt)
            prompt_name: Name for this prompt (for logging)

        Returns:
            Dictionary of evaluation metrics

        Example:
            >>> prompt_pair = ("cancer", "normal")
            >>> metrics = evaluator.evaluate(dataset, prompt_pair, "minimal_1")
            >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Get text prompts
        positive_prompt, negative_prompt = prompt_pair
        text_prompts = [positive_prompt, negative_prompt]

        # Encode text prompts once
        with torch.no_grad():
            text_features = self.model.encode_text(text_prompts)

        # Collect predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Evaluate
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {prompt_name}", leave=False):
                # Unpack batch
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch

                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get predictions
                predictions, probabilities = self.model.zero_shot_predict(
                    images, text_prompts
                )

                # Collect results
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        # Concatenate all batches
        y_pred = np.concatenate(all_predictions)
        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_probabilities)

        # Compute metrics
        num_classes = len(np.unique(y_true))
        task = 'binary' if num_classes == 2 else 'multiclass'

        metrics_calculator = MetricsCalculator(task=task, num_classes=num_classes)
        metrics = metrics_calculator.compute(y_true, y_pred, y_proba)

        # Add prompt information
        metrics['prompt_name'] = prompt_name
        metrics['prompt_positive'] = positive_prompt
        metrics['prompt_negative'] = negative_prompt

        return metrics

    def evaluate_all_prompts(
        self,
        dataset,
        prompt_manager: PromptManager,
        save_results: bool = True,
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model with all prompts from prompt manager.

        Args:
            dataset: Dataset to evaluate on
            prompt_manager: PromptManager with prompts to test
            save_results: Whether to save results to disk
            save_dir: Directory to save results (default: results/zero_shot)

        Returns:
            Dictionary mapping prompt names to metrics

        Example:
            >>> prompt_manager = PromptManager('config/prompt_configs.yaml')
            >>> results = evaluator.evaluate_all_prompts(dataset, prompt_manager)
            >>> print(f"Evaluated {len(results)} prompts")
        """
        all_results = {}

        # Get all prompts
        all_prompts = prompt_manager.get_all_prompts()

        print(f"Evaluating {len(all_prompts)} prompts...")

        # Evaluate each prompt
        for idx, (category, positive, negative) in enumerate(tqdm(all_prompts, desc="Prompts")):
            # Create prompt name - use enumeration to avoid list.index() issues
            category_prompts = prompt_manager.get_prompts_by_category(category)
            # Find which prompt this is within the category
            prompt_idx = None
            for i, (pos, neg) in enumerate(category_prompts):
                if pos == positive and neg == negative:
                    prompt_idx = i
                    break

            if prompt_idx is None:
                prompt_idx = 0  # Fallback

            prompt_name = f"{category}_{prompt_idx + 1}"

            # Evaluate
            metrics = self.evaluate(dataset, (positive, negative), prompt_name)
            all_results[prompt_name] = metrics

            # Print metrics immediately after evaluation
            print(f"\n{prompt_name:20s} | Acc: {metrics.get('accuracy', 0.0):.4f} | "
                  f"F1: {metrics.get('f1', 0.0):.4f} | AUC: {metrics.get('auc_roc', 0.0):.4f}")
            print(f"  Prompts: ['{negative}', '{positive}']")

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(all_results)
        all_results['_aggregate'] = aggregate_stats

        # Save results
        if save_results:
            self._save_results(all_results, save_dir)

        return all_results

    def evaluate_by_category(
        self,
        dataset,
        prompt_manager: PromptManager,
        category: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model with prompts from a specific category.

        Args:
            dataset: Dataset to evaluate on
            prompt_manager: PromptManager instance
            category: Category name

        Returns:
            Dictionary mapping prompt names to metrics

        Example:
            >>> results = evaluator.evaluate_by_category(
            ...     dataset, prompt_manager, 'minimal'
            ... )
        """
        category_results = {}

        # Get prompts for this category
        prompts = prompt_manager.get_prompts_by_category(category)

        for idx, (positive, negative) in enumerate(prompts):
            prompt_name = f"{category}_{idx + 1}"
            metrics = self.evaluate(dataset, (positive, negative), prompt_name)
            category_results[prompt_name] = metrics

        return category_results

    def _compute_aggregate_statistics(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute aggregate statistics across all prompts."""
        # Filter out metadata keys
        metric_results = {
            k: v for k, v in results.items()
            if not k.startswith('_')
        }

        # Extract numeric metrics only
        numeric_metrics = {}
        for prompt_name, metrics in metric_results.items():
            numeric_metrics[prompt_name] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not k.startswith('prompt_')
            }

        # Compute statistics
        aggregate = {}
        if numeric_metrics:
            metrics_list = list(numeric_metrics.values())
            aggregate = aggregate_metrics_over_runs(metrics_list)

            # Add prompt sensitivity analysis
            for metric_name in ['accuracy', 'f1', 'balanced_accuracy']:
                if any(metric_name in m for m in metrics_list):
                    sensitivity = analyze_prompt_sensitivity(numeric_metrics, metric_name)
                    for stat_name, value in sensitivity.items():
                        aggregate[f'{metric_name}_sensitivity_{stat_name}'] = {
                            'mean': value, 'std': 0.0, 'min': value, 'max': value,
                            'ci_lower': value, 'ci_upper': value
                        }

        return aggregate

    def _save_results(
        self,
        results: Dict,
        save_dir: Optional[str] = None
    ):
        """Save evaluation results to disk."""
        if save_dir is None:
            save_dir = Path('results/zero_shot')
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Get model name
        model_name = self.model.__class__.__name__.lower().replace('model', '')

        # Save full results as pickle
        results_file = save_dir / f'{model_name}_zero_shot_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to: {results_file}")

        # Save summary as text
        summary_file = save_dir / f'{model_name}_zero_shot_summary.txt'
        with open(summary_file, 'w') as f:
            self._write_summary(results, f)

        print(f"Summary saved to: {summary_file}")

    def _write_summary(self, results: Dict, file):
        """Write results summary to file."""
        file.write("=" * 80 + "\n")
        file.write("ZERO-SHOT EVALUATION SUMMARY\n")
        file.write("=" * 80 + "\n\n")

        # Model info
        model_info = self.model.get_model_info()
        file.write("Model Information:\n")
        file.write(f"  Name: {model_info['model_name']}\n")
        file.write(f"  Device: {model_info['device']}\n")
        file.write(f"  Parameters: {model_info['parameters']['total']:,}\n\n")

        # Aggregate statistics
        if '_aggregate' in results:
            file.write("Aggregate Statistics Across All Prompts:\n")
            file.write("-" * 80 + "\n")

            aggregate = results['_aggregate']
            for metric_name in ['accuracy', 'f1', 'balanced_accuracy']:
                if metric_name in aggregate:
                    stats = aggregate[metric_name]
                    file.write(
                        f"{metric_name:25s}: "
                        f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                        f"[{stats['min']:.4f}, {stats['max']:.4f}]\n"
                    )
            file.write("\n")

        # Best prompts
        file.write("Best Performing Prompts:\n")
        file.write("-" * 80 + "\n")

        metric_results = {
            k: v for k, v in results.items()
            if not k.startswith('_')
        }

        for metric_name in ['accuracy', 'f1']:
            try:
                best_prompt, best_value = find_best_prompt(metric_results, metric_name)
                file.write(f"  {metric_name}: {best_prompt} ({best_value:.4f})\n")
            except:
                pass

        file.write("\n")

        # Per-prompt results
        file.write("Per-Prompt Results:\n")
        file.write("-" * 80 + "\n")
        file.write(f"{'Prompt':<20s} {'Accuracy':>10s} {'F1':>10s} {'AUC-ROC':>10s}\n")
        file.write("-" * 80 + "\n")

        for prompt_name, metrics in sorted(metric_results.items()):
            acc = metrics.get('accuracy', 0.0)
            f1 = metrics.get('f1', 0.0)
            auc = metrics.get('auc_roc', 0.0)
            file.write(f"{prompt_name:<20s} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}\n")

    def get_best_prompt(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'accuracy'
    ) -> Tuple[str, float]:
        """
        Get the best performing prompt from results.

        Args:
            results: Results dictionary from evaluate_all_prompts
            metric: Metric to use for comparison

        Returns:
            Tuple of (best_prompt_name, best_metric_value)

        Example:
            >>> results = evaluator.evaluate_all_prompts(dataset, prompt_manager)
            >>> best_name, best_value = evaluator.get_best_prompt(results)
            >>> print(f"Best prompt: {best_name} ({best_value:.4f})")
        """
        # Filter out aggregate stats
        metric_results = {
            k: v for k, v in results.items()
            if not k.startswith('_')
        }
        return find_best_prompt(metric_results, metric)
