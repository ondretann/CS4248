"""
Ensemble methods for combining multiple QA models.
"""

import numpy as np
from typing import Dict, List, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class QAEnsemble:
    """
    Ensemble multiple QA models for improved predictions.
    """
    
    def __init__(
        self,
        models: List[PreTrainedModel],
        tokenizers: List[PreTrainedTokenizer],
        weights: List[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ensemble.
        
        Args:
            models: List of trained QA models
            tokenizers: List of corresponding tokenizers
            weights: Optional weights for each model (defaults to uniform)
            device: Device to run inference on
        """
        self.models = models
        self.tokenizers = tokenizers
        self.device = device
        
        # Set uniform weights if not provided
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        # Move models to device
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict_logits_averaging(
        self,
        question: str,
        context: str,
        max_length: int = 384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using logits averaging ensemble.
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (averaged_start_logits, averaged_end_logits)
        """
        all_start_logits = []
        all_end_logits = []
        
        for model, tokenizer, weight in zip(self.models, self.tokenizers, self.weights):
            # Tokenize
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=max_length,
                truncation="only_second",
                padding="max_length"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Weight the logits
            start_logits = outputs.start_logits[0].cpu().numpy() * weight
            end_logits = outputs.end_logits[0].cpu().numpy() * weight
            
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
        
        # Average logits
        avg_start_logits = np.sum(all_start_logits, axis=0)
        avg_end_logits = np.sum(all_end_logits, axis=0)
        
        return avg_start_logits, avg_end_logits
    
    def predict_probability_averaging(
        self,
        question: str,
        context: str,
        max_length: int = 384
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using probability averaging ensemble.
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (averaged_start_probs, averaged_end_probs)
        """
        all_start_probs = []
        all_end_probs = []
        
        for model, tokenizer, weight in zip(self.models, self.tokenizers, self.weights):
            # Tokenize
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=max_length,
                truncation="only_second",
                padding="max_length"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Convert to probabilities and weight
            start_probs = torch.softmax(outputs.start_logits[0], dim=0).cpu().numpy() * weight
            end_probs = torch.softmax(outputs.end_logits[0], dim=0).cpu().numpy() * weight
            
            all_start_probs.append(start_probs)
            all_end_probs.append(end_probs)
        
        # Average probabilities
        avg_start_probs = np.sum(all_start_probs, axis=0)
        avg_end_probs = np.sum(all_end_probs, axis=0)
        
        return avg_start_probs, avg_end_probs
    
    def predict_voting(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        n_best: int = 5
    ) -> str:
        """
        Predict using voting ensemble (select most common answer).
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum sequence length
            n_best: Number of best answers to consider from each model
            
        Returns:
            Most voted answer string
        """
        all_answers = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
            # Tokenize
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=max_length,
                truncation="only_second",
                padding="max_length"
            )
            inputs_dict = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs_dict)
            
            # Get top-k start and end positions
            start_logits = outputs.start_logits[0].cpu()
            end_logits = outputs.end_logits[0].cpu()
            
            start_indexes = torch.argsort(start_logits, descending=True)[:n_best]
            end_indexes = torch.argsort(end_logits, descending=True)[:n_best]
            
            # Extract answers
            for start_idx in start_indexes:
                for end_idx in end_indexes:
                    if end_idx >= start_idx:
                        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
                        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                        all_answers.append(answer)
        
        # Vote for most common answer
        if not all_answers:
            return ""
        
        # Count occurrences
        from collections import Counter
        answer_counts = Counter(all_answers)
        most_common = answer_counts.most_common(1)[0][0]
        
        return most_common


def load_ensemble(
    model_paths: List[str],
    weights: List[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> QAEnsemble:
    """
    Load an ensemble of models from paths.
    
    Args:
        model_paths: List of paths to trained models
        weights: Optional weights for each model
        device: Device to run inference on
        
    Returns:
        QAEnsemble object
    """
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    
    models = []
    tokenizers = []
    
    print("Loading ensemble models...")
    for path in model_paths:
        print(f"  Loading: {path}")
        model = AutoModelForQuestionAnswering.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        models.append(model)
        tokenizers.append(tokenizer)
    
    print(f"Ensemble loaded with {len(models)} models")
    
    return QAEnsemble(models, tokenizers, weights, device)


if __name__ == "__main__":
    # Example usage
    print("Ensemble module for combining multiple QA models")
    print("Use load_ensemble() to create an ensemble from model paths")




