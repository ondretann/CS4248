"""Training script for QA model."""

import os, sys, argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator

sys.path.append(str(Path(__file__).parent.parent))
from src.data_preprocessing import load_and_prepare_data
from src.evaluation import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train QA model")
    
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", help="Model name")
    parser.add_argument("--train_file", required=True, help="Training JSON file")
    parser.add_argument("--validation_file", required=True, help="Validation JSON file")
    parser.add_argument("--max_length", type=int, default=384, help="Max sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Doc stride")
    parser.add_argument("--output_dir", default="./models/deberta-v3-squad", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=24, help="Batch size (increase if you have memory)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=24, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=200, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluate every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check device and disable fp16 for MPS (Apple Silicon doesn't support it)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and args.fp16:
        print("⚠️  Warning: fp16 not supported on MPS (Apple Silicon). Disabling fp16.")
        args.fp16 = False
    
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nDevice: {device}")
    print(f"Training: {args.model_name}")
    print(f"Data: {args.train_file}, {args.validation_file}")
    print(f"Output: {args.output_dir}")
    print(f"\nTraining config:")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Batch size: {args.per_device_train_batch_size} (effective: {args.per_device_train_batch_size * args.gradient_accumulation_steps})")
    print(f"  Eval every: {args.eval_steps} steps")
    print(f"  Save every: {args.save_steps} steps\n")
    
    train_dataset, eval_dataset, raw_eval_dataset, tokenizer = load_and_prepare_data(
        args.train_file, args.validation_file, args.model_name, 
        args.max_length, args.doc_stride, args.num_proc)
    
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    
    # Explicitly move model to device (MPS/CUDA/CPU)
    if device == "mps":
        model = model.to("mps")
        print(f"✓ Model moved to MPS (Apple Silicon GPU)")
    elif device == "cuda":
        model = model.to("cuda")
        print(f"✓ Model moved to CUDA")
    else:
        print(f"Using CPU")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        eval_strategy="no",  # No mid-training eval for speed
        save_steps=args.save_steps, 
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs, 
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, 
        fp16=args.fp16, 
        logging_steps=args.logging_steps,
        report_to=[], 
        dataloader_pin_memory=False,
        dataloader_num_workers=0  # Single worker to avoid serialization issues
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )
    
    print("\nTraining...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\nEvaluating...")
    eval_metrics, _ = evaluate_model(model, tokenizer, eval_dataset, raw_eval_dataset, 
                                     args.per_device_eval_batch_size)
    
    print(f"\nResults: EM={eval_metrics['exact_match']:.2f}%, F1={eval_metrics['f1']:.2f}%")
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    
    print(f"Done! Saved to {args.output_dir}")
    

if __name__ == "__main__":
    main()

