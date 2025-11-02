"""
Inference script for making predictions with a trained QA model.
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


def predict_answer(
    question: str,
    context: str,
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    device: str = "cpu"
) -> dict:
    """
    Predict an answer for a given question and context.
    
    Args:
        question: Question text
        context: Context text
        model: Trained QA model
        tokenizer: Tokenizer
        device: Device to run inference on
        
    Returns:
        Dictionary with answer, start/end positions, and confidence score
    """
    # Tokenize input
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation="only_second",
        padding="max_length"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get start and end positions
    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()
    
    # Get confidence score
    start_score = torch.softmax(start_logits, dim=0)[start_idx].item()
    end_score = torch.softmax(end_logits, dim=0)[end_idx].item()
    confidence = (start_score + end_score) / 2
    
    # Decode answer
    if end_idx >= start_idx:
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    else:
        answer = ""
    
    return {
        "answer": answer,
        "start_position": start_idx,
        "end_position": end_idx,
        "confidence": confidence
    }


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Make predictions with a trained QA model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question text"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context text"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="JSON file with questions and contexts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.json",
        help="Output file for predictions"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Single prediction mode
    if args.question and args.context:
        print("\n" + "="*50)
        print("SINGLE PREDICTION")
        print("="*50)
        print(f"\nQuestion: {args.question}")
        print(f"Context: {args.context[:200]}..." if len(args.context) > 200 else f"Context: {args.context}")
        
        result = predict_answer(args.question, args.context, model, tokenizer, device)
        
        print(f"\n{'='*50}")
        print("RESULT")
        print("="*50)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("="*50 + "\n")
        
    # Batch prediction mode
    elif args.input_file:
        print(f"\nLoading questions from: {args.input_file}")
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        results = []
        print(f"\nProcessing {len(data)} examples...")
        
        for i, example in enumerate(data):
            question = example.get('question', '')
            context = example.get('context', '')
            
            result = predict_answer(question, context, model, tokenizer, device)
            result['id'] = example.get('id', i)
            result['question'] = question
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)} examples")
        
        # Save results
        print(f"\nSaving predictions to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Done!")
    
    else:
        print("Error: Please provide either --question and --context, or --input_file")


if __name__ == "__main__":
    main()


