"""
Run zero-shot evaluation on histopathology datasets.

This script evaluates vision-language models in a zero-shot setting
using multiple prompt variations.
"""

import argparse
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import PCamDataset
from src.evaluation import ZeroShotEvaluator, PromptManager
from src.utils import set_seed, setup_logger, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run zero-shot evaluation on histopathology datasets'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='conch',
        choices=['conch', 'pathgen_clip', 'keep', 'quiltnet', 'plip', 'llava_med'],
        help='Model to evaluate'
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default='config/model_configs.yaml',
        help='Path to model configuration file'
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='pcam',
        choices=['pcam', 'breakhis', 'bach', 'idc'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/PCam',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'valid', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--subset_fraction',
        type=float,
        default=None,
        help='Fraction of dataset to use (for quick testing)'
    )

    # Prompt arguments
    parser.add_argument(
        '--prompt_config',
        type=str,
        default='config/prompt_configs.yaml',
        help='Path to prompt configuration file'
    )
    parser.add_argument(
        '--prompt_category',
        type=str,
        default=None,
        help='Specific prompt category to evaluate (None = all)'
    )

    # Evaluation arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, None=auto)'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/zero_shot',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        default=True,
        help='Save results to disk'
    )

    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token for model access'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(
        'zero_shot_eval',
        log_file=f'logs/zero_shot_{args.model}_{args.dataset}.log'
    )

    logger.info("=" * 80)
    logger.info("ZERO-SHOT EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset} ({args.split} split)")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device or 'auto'}")
    logger.info("=" * 80)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Set HuggingFace token if provided
    if args.hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = args.hf_token
        logger.info("HuggingFace token set")

    try:
        # Load model
        logger.info(f"Loading model: {args.model}...")
        model = create_model(
            args.model,
            config_path=args.model_config,
            device=args.device or 'cuda'
        )
        logger.info(f"Model loaded successfully")
        model_info = model.get_model_info()
        logger.info(f"Total parameters: {model_info['parameters']['total']:,}")

        # Load dataset
        logger.info(f"Loading dataset: {args.dataset}...")
        if args.dataset == 'pcam':
            # Get transform from model
            transform = model.get_preprocessor()

            dataset = PCamDataset(
                data_dir=args.data_dir,
                split=args.split,
                transform=transform,
                load_into_memory=False  # Use on-demand loading for large dataset
            )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not yet implemented")

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # Create subset if requested
        if args.subset_fraction is not None:
            logger.info(f"Creating {args.subset_fraction*100:.1f}% subset...")
            dataset = dataset.get_subset(
                fraction=args.subset_fraction,
                seed=args.seed,
                stratified=True
            )
            logger.info(f"Subset created: {len(dataset)} samples")

        # Load prompts
        logger.info(f"Loading prompts from: {args.prompt_config}")
        prompt_manager = PromptManager(args.prompt_config)
        prompt_stats = prompt_manager.get_prompt_statistics()
        logger.info(f"Loaded {prompt_stats['total']} prompts across {len(prompt_stats)-1} categories")

        # Create evaluator
        logger.info("Creating zero-shot evaluator...")
        evaluator = ZeroShotEvaluator(
            model=model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        if args.prompt_category:
            # Evaluate specific category
            logger.info(f"Evaluating category: {args.prompt_category}")
            results = evaluator.evaluate_by_category(
                dataset, prompt_manager, args.prompt_category
            )
        else:
            # Evaluate all prompts
            results = evaluator.evaluate_all_prompts(
                dataset,
                prompt_manager,
                save_results=args.save_results,
                save_dir=args.output_dir
            )

        # Print summary
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)

        # Get aggregate stats
        if '_aggregate' in results:
            aggregate = results['_aggregate']
            logger.info("\nAggregate Statistics:")
            for metric_name in ['accuracy', 'f1', 'balanced_accuracy']:
                if metric_name in aggregate:
                    stats = aggregate[metric_name]
                    logger.info(
                        f"  {metric_name}: "
                        f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                        f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                    )

        # Find best prompts
        logger.info("\nBest Performing Prompts:")
        metric_results = {
            k: v for k, v in results.items() if not k.startswith('_')
        }
        for metric in ['accuracy', 'f1']:
            try:
                best_prompt, best_value = evaluator.get_best_prompt(
                    results, metric
                )
                logger.info(f"  {metric}: {best_prompt} ({best_value:.4f})")
            except:
                pass

        logger.info("=" * 80)

        if args.save_results:
            logger.info(f"\nResults saved to: {args.output_dir}")

        logger.info("\nEvaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
