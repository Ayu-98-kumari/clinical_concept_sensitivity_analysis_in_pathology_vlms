"""
Simple example demonstrating zero-shot evaluation.

This script shows how to use the framework for zero-shot evaluation
with minimal code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import create_model
from src.data import PCamDataset
from src.evaluation import ZeroShotEvaluator, PromptManager
from src.utils import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(42)

    print("=" * 80)
    print("ZERO-SHOT EVALUATION EXAMPLE")
    print("=" * 80)

    # 1. Load model
    print("\n1. Loading CONCH model...")
    model = create_model(
        'conch',
        config_path='config/model_configs.yaml',
        device='cuda'
    )
    print(f"   Model loaded: {model.__class__.__name__}")
    print(f"   Parameters: {model.count_parameters()['total']:,}")

    # 2. Load dataset (use a small subset for quick testing)
    print("\n2. Loading PCam dataset (test split)...")
    transform = model.get_preprocessor()
    dataset = PCamDataset(
        data_dir="data/PCam",
        split="test",
        transform=transform,
        load_into_memory=False
    )
    print(f"   Dataset size: {len(dataset)} samples")

    # Create a small subset for quick testing
    print("   Creating 1% subset for quick testing...")
    dataset_subset = dataset.get_subset(fraction=0.01, seed=42)
    print(f"   Subset size: {len(dataset_subset)} samples")

    # 3. Load prompts
    print("\n3. Loading prompts...")
    prompt_manager = PromptManager('config/prompt_configs.yaml')
    stats = prompt_manager.get_prompt_statistics()
    print(f"   Total prompts: {stats['total']}")
    for category in prompt_manager.get_categories():
        print(f"     - {category}: {stats[category]} prompts")

    # 4. Create evaluator
    print("\n4. Creating zero-shot evaluator...")
    evaluator = ZeroShotEvaluator(
        model=model,
        batch_size=32,
        num_workers=4
    )
    print("   Evaluator created")

    # 5. Evaluate with a single prompt
    print("\n5. Evaluating with a single prompt...")
    prompt_pair = ("cancer", "normal")
    metrics = evaluator.evaluate(dataset_subset, prompt_pair, "minimal_1")

    print("\n   Results:")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     F1 Score: {metrics['f1']:.4f}")
    print(f"     Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    if 'auc_roc' in metrics:
        print(f"     AUC-ROC: {metrics['auc_roc']:.4f}")

    # 6. Evaluate with all prompts (optional - takes longer)
    print("\n6. Would you like to evaluate all 20 prompts? (y/n): ", end='')
    response = input().strip().lower()

    if response == 'y':
        print("\n   Evaluating all 20 prompts...")
        results = evaluator.evaluate_all_prompts(
            dataset_subset,
            prompt_manager,
            save_results=True,
            save_dir='results/zero_shot'
        )

        # Show summary
        if '_aggregate' in results:
            print("\n   Aggregate Results:")
            aggregate = results['_aggregate']
            for metric_name in ['accuracy', 'f1']:
                if metric_name in aggregate:
                    stats = aggregate[metric_name]
                    print(
                        f"     {metric_name}: "
                        f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                    )

        # Best prompt
        best_prompt, best_acc = evaluator.get_best_prompt(results, 'accuracy')
        print(f"\n   Best prompt: {best_prompt} (Accuracy: {best_acc:.4f})")

        print("\n   Results saved to: results/zero_shot/")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
