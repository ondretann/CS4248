"""
Evaluation utilities for extractive question answering.
Computes metrics like Exact Match (EM) and F1 score.
"""

import collections
import string
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from tqdm.auto import tqdm


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison.
    
    Args:
        s: Answer string
        
    Returns:
        Normalized answer string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match score.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # If either is empty
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common_tokens = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_metrics(predictions: Dict[str, str], references: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute EM and F1 scores for all predictions.
    
    Args:
        predictions: Dictionary mapping example IDs to predicted answers
        references: Dictionary mapping example IDs to list of ground truth answers
        
    Returns:
        Dictionary with 'exact_match' and 'f1' scores
    """
    exact_match_scores = []
    f1_scores = []
    
    for example_id, prediction in predictions.items():
        if example_id not in references:
            continue
            
        ground_truths = references[example_id]
        
        # Compute max scores across all ground truths
        em_score = max(compute_exact_match(prediction, gt) for gt in ground_truths)
        f1_score = max(compute_f1(prediction, gt) for gt in ground_truths)
        
        exact_match_scores.append(em_score)
        f1_scores.append(f1_score)
    
    return {
        'exact_match': np.mean(exact_match_scores) * 100,
        'f1': np.mean(f1_scores) * 100,
        'total': len(exact_match_scores)
    }


def postprocess_qa_predictions(
    examples: Dataset,
    features: Dataset,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: Optional[float] = 0.0
) -> Dict[str, str]:
    """
    Post-process model predictions to extract final answers.
    
    Args:
        examples: Raw validation examples
        features: Tokenized validation features
        predictions: Tuple of (start_logits, end_logits)
        n_best_size: Number of n-best predictions to consider
        max_answer_length: Maximum length of predicted answer
        null_score_diff_threshold: Threshold for null answer (for SQuAD 2.0)
        
    Returns:
        Dictionary mapping example IDs to predicted answers
    """
    start_logits, end_logits = predictions
    
    # Build a map from example to features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    # Loop through all examples
    predictions_dict = {}
    
    for example_index, example in enumerate(tqdm(examples, desc="Post-processing")):
        feature_indices = features_per_example[example_index]
        
        min_null_score = None
        prelim_predictions = []
        
        # Loop through all features associated with the example
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            # Update minimum null prediction
            # CLS token is always at position 0 (first token)
            cls_index = 0
            feature_null_score = start_logit[cls_index] + end_logit[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score
            
            # Get valid start and end positions
            start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid predictions
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logit[start_index] + end_logit[end_index],
                            "start_logit": start_logit[start_index],
                            "end_logit": end_logit[end_index],
                        }
                    )
        
        # Only keep the best n_best_size predictions
        predictions_sorted = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)
        predictions_sorted = predictions_sorted[:n_best_size]
        
        # Get the best non-null prediction
        if len(predictions_sorted) > 0:
            best_pred = predictions_sorted[0]
            context = example["context"]
            pred_text = context[best_pred["offsets"][0]:best_pred["offsets"][1]]
            
            # Check if we should predict null answer (for SQuAD 2.0)
            if null_score_diff_threshold is not None:
                score_diff = min_null_score - best_pred["score"]
                if score_diff > null_score_diff_threshold:
                    pred_text = ""
        else:
            pred_text = ""
        
        predictions_dict[example["id"]] = pred_text
    
    return predictions_dict


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    raw_dataset: Dataset,
    batch_size: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Evaluate a model on a validation dataset.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        eval_dataset: Tokenized evaluation dataset
        raw_dataset: Raw evaluation dataset
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Tuple of (metrics, predictions)
    """
    model.eval()
    model.to(device)
    
    # Collect predictions
    all_start_logits = []
    all_end_logits = []
    
    print("Generating predictions...")
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        batch = eval_dataset[i:i + batch_size]
        
        # Prepare batch
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }
        
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = torch.tensor(batch["token_type_ids"]).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    # Concatenate all predictions
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)
    
    # Post-process predictions
    predictions = postprocess_qa_predictions(
        examples=raw_dataset,
        features=eval_dataset,
        predictions=(start_logits, end_logits)
    )
    
    # Prepare references
    references = {}
    for example in raw_dataset:
        if len(example["answers"]["text"]) > 0:
            references[example["id"]] = example["answers"]["text"]
        else:
            references[example["id"]] = [""]
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    
    return metrics, predictions


if __name__ == "__main__":
    # Example usage
    predictions = {
        "example1": "The answer",
        "example2": "Another answer"
    }
    
    references = {
        "example1": ["The answer", "the answer"],
        "example2": ["Another answer"]
    }
    
    metrics = compute_metrics(predictions, references)
    print(f"Exact Match: {metrics['exact_match']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")

