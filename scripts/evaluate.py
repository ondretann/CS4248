"""
Evaluation script for trained QA models.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import load_and_prepare_data
from src.evaluation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained QA model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        required=True,
        help="Path to validation JSON file (e.g., dev-v1.1.json)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="Document stride"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="File to save predictions"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("\n" + "="*50)
    print("EVALUATION CONFIGURATION")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Validation file: {args.validation_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_length}")
    print("="*50 + "\n")
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    
    # Load validation data only (no need for training data)
    print("\nLoading and preparing validation data...")
    from src.data_preprocessing import load_qa_dataset_from_files, prepare_datasets
    
    # Create a dummy train file path (we won't use it)
    # But the function expects both files
    dataset = load_qa_dataset_from_files(
        train_file=args.validation_file,  # Use validation as dummy
        validation_file=args.validation_file
    )
    
    # Prepare only validation dataset
    _, eval_dataset, raw_eval_dataset = prepare_datasets(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        num_proc=4
    )
    
    # Subset if requested
    if args.num_examples is not None:
        print(f"\nEvaluating on {args.num_examples} examples")
        eval_dataset = eval_dataset.select(range(args.num_examples))
        raw_eval_dataset = raw_eval_dataset.select(range(args.num_examples))
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Evaluate
    print("\n" + "="*50)
    print("STARTING EVALUATION")
    print("="*50 + "\n")
    
    eval_metrics, predictions = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        raw_dataset=raw_eval_dataset,
        batch_size=args.batch_size,
        device=device
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Exact Match: {eval_metrics['exact_match']:.2f}%")
    print(f"F1 Score: {eval_metrics['f1']:.2f}%")
    print(f"Total examples: {eval_metrics['total']}")
    print("="*50 + "\n")
    
    # Save results
    if args.output_file:
        print(f"Saving results to: {args.output_file}")
        results = {
            "model_path": args.model_path,
            "dataset": args.dataset_name,
            "metrics": eval_metrics,
            "config": {
                "max_length": args.max_length,
                "doc_stride": args.doc_stride,
                "batch_size": args.batch_size
            }
        }
        
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    # Save predictions
    if args.save_predictions:
        print(f"Saving predictions to: {args.save_predictions}")
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        with open(args.save_predictions, "w") as f:
            json.dump(predictions, f, indent=2)
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 80)
    for i, example in enumerate(raw_eval_dataset.select(range(min(3, len(raw_eval_dataset))))):
        example_id = example["id"]
        question = example["question"]
        true_answer = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
        predicted_answer = predictions.get(example_id, "")
        
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted: {predicted_answer}")
        print(f"Match: {'✓' if predicted_answer.lower() == true_answer.lower() else '✗'}")
    print("-" * 80)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()


