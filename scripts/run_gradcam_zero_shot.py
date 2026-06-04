"""
Run GradCAM visualization for zero-shot evaluation.

This script generates GradCAM visualizations for all models and prompts
on a subset of the PCam dataset to understand which regions are important
for zero-shot predictions.
"""

import argparse
import sys
from pathlib import Path
import os
from tqdm import tqdm
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import PCamDataset
from src.evaluation import PromptManager
from src.visualization import GradCAM, create_gradcam_grid
from src.utils import set_seed, setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate GradCAM visualizations for zero-shot evaluation'
    )

    # Model arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['conch', 'pathgen_clip', 'keep', 'quiltnet', 'plip'],
        help='Models to visualize'
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default='config/model_configs.yaml',
        help='Path to model configuration file'
    )

    # Dataset arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/PCam',
        help='Path to PCam dataset directory'
    )
    parser.add_argument(
        '--subset_fraction',
        type=float,
        default=0.1,
        help='Fraction of dataset to visualize (default: 0.1 = 10%)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of samples to visualize per model-prompt combo'
    )

    # Prompt arguments
    parser.add_argument(
        '--prompt_config',
        type=str,
        default='config/prompt_configs.yaml',
        help='Path to prompt configuration file'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/gradcam_zero_shot',
        help='Directory to save visualizations'
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
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set up logging
    logger = setup_logger(
        'gradcam_zero_shot',
        log_file='logs/gradcam_zero_shot.log'
    )

    logger.info("=" * 80)
    logger.info("GRADCAM VISUALIZATION FOR ZERO-SHOT EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Subset fraction: {args.subset_fraction * 100:.1f}%")
    logger.info(f"Samples per combo: {args.num_samples}")
    logger.info("=" * 80)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Set HuggingFace token if provided
    if args.hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = args.hf_token
        logger.info("HuggingFace token set")

    try:
        # Load prompts
        logger.info(f"Loading prompts from: {args.prompt_config}")
        prompt_manager = PromptManager(args.prompt_config)
        all_prompts = prompt_manager.get_all_prompts()
        logger.info(f"Loaded {len(all_prompts)} prompts")

        # Process each model
        for model_name in args.models:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'=' * 80}")

            # Load model
            logger.info(f"Loading model: {model_name}...")
            try:
                model = create_model(
                    model_name,
                    config_path=args.model_config,
                    device=args.device
                )
                logger.info(f"Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue

            # Load dataset with model-specific preprocessing
            logger.info("Loading dataset...")
            transform = model.get_preprocessor()
            dataset = PCamDataset(
                data_dir=args.data_dir,
                split='test',
                transform=transform,
                load_into_memory=False
            )

            # Create subset
            logger.info(f"Creating {args.subset_fraction*100:.1f}% subset...")
            dataset = dataset.get_subset(
                fraction=args.subset_fraction,
                seed=args.seed,
                stratified=True
            )
            logger.info(f"Subset created: {len(dataset)} samples")

            # Sample indices for visualization
            num_vis_samples = min(args.num_samples, len(dataset))
            vis_indices = np.random.choice(
                len(dataset),
                size=num_vis_samples,
                replace=False
            )
            logger.info(f"Selected {num_vis_samples} samples for visualization")

            # Process each prompt
            for prompt_idx, (category, positive, negative) in enumerate(tqdm(all_prompts, desc="Prompts")):
                # Create prompt name
                category_prompts = prompt_manager.get_prompts_by_category(category)
                prompt_idx_in_cat = None
                for i, (pos, neg) in enumerate(category_prompts):
                    if pos == positive and neg == negative:
                        prompt_idx_in_cat = i
                        break
                prompt_name = f"{category}_{prompt_idx_in_cat + 1 if prompt_idx_in_cat is not None else 0}"

                text_prompts = [negative, positive]  # Order: [normal, tumor]

                logger.info(f"  Processing prompt: {prompt_name}")
                logger.info(f"    Prompts: {text_prompts}")

                # Create output directory
                output_dir = Path(args.output_dir) / model_name / prompt_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate visualizations
                visualizations = []
                titles = []

                for i, idx in enumerate(tqdm(vis_indices, desc=f"    {prompt_name}", leave=False)):
                    # Get sample
                    image, label, metadata = dataset[idx]
                    image = image.unsqueeze(0).to(args.device)

                    # Get prediction
                    with torch.no_grad():
                        predictions, probabilities = model.zero_shot_predict(
                            image, text_prompts
                        )
                        pred_class = predictions[0].item()
                        confidence = probabilities[0, pred_class].item()

                    # Generate GradCAM
                    try:
                        # Create fresh GradCAM object for each sample to avoid hook issues
                        gradcam = GradCAM(model)
                        cam = gradcam.generate(image, text_prompts, target_class=pred_class)

                        # Create visualization
                        vis = gradcam.visualize(image, cam, alpha=0.4)
                        visualizations.append(vis)

                        # Create title
                        pred_label = text_prompts[pred_class]
                        true_label = text_prompts[label]
                        correct = "✓" if pred_class == label else "✗"
                        title = f"ID:{idx} {pred_label}({confidence:.2f}) vs {true_label} {correct}"
                        titles.append(title)

                        # Denormalize image for display
                        img_np = image.squeeze().cpu().numpy()
                        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                        img_np = img_np * std + mean
                        img_np = np.clip(img_np, 0, 1)
                        img_np = np.transpose(img_np, (1, 2, 0))

                        # Save individual visualization with 3-panel view
                        save_path = output_dir / f"sample_{idx}.png"
                        gradcam.save_visualization(
                            vis, str(save_path), title=title,
                            original_image=img_np, heatmap=cam
                        )

                    except Exception as e:
                        logger.warning(f"    Failed to generate GradCAM for sample {idx}: {e}")
                        continue

                # Create grid visualization
                if visualizations:
                    grid_path = output_dir / f"{prompt_name}_grid.png"
                    try:
                        create_gradcam_grid(
                            visualizations[:25],  # Limit to 25 for grid
                            str(grid_path),
                            titles=titles[:25],
                            grid_size=(5, 5)
                        )
                        logger.info(f"    Saved grid visualization: {grid_path}")
                    except Exception as e:
                        logger.warning(f"    Failed to create grid: {e}")

            logger.info(f"Completed visualization for {model_name}")

        logger.info("\n" + "=" * 80)
        logger.info("ALL GRADCAM VISUALIZATIONS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
