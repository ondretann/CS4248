"""Data preprocessing for extractive QA."""

from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer


class QADataPreprocessor:
    """Preprocessor for QA datasets."""
    
    def __init__(self, tokenizer, max_length=384, doc_stride=128, pad_on_right=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.pad_on_right = pad_on_right
    
    def prepare_train_features(self, examples: Dict) -> Dict:
        """Tokenize training examples."""
        # Tokenize
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1
                
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1
                
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples
    
    def prepare_validation_features(self, examples: Dict) -> Dict:
        """Tokenize validation examples."""
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples


def load_qa_dataset_from_files(train_file: str, validation_file: str) -> DatasetDict:
    """Load QA dataset from local JSON files."""
    print(f"Loading: {train_file}, {validation_file}")
    
    dataset = load_dataset('json', data_files={'train': train_file, 'validation': validation_file}, field='data')
    
    # Flatten SQuAD JSON structure
    processed_dataset = {}
    for split in dataset.keys():
        flat_data = {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []}
        
        for example in dataset[split]:
            for paragraph in example['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    flat_data['id'].append(qa['id'])
                    flat_data['title'].append(example.get('title', ''))
                    flat_data['context'].append(context)
                    flat_data['question'].append(qa['question'])
                    flat_data['answers'].append({
                        'text': [ans['text'] for ans in qa['answers']],
                        'answer_start': [ans['answer_start'] for ans in qa['answers']]
                    } if qa['answers'] else {'text': [], 'answer_start': []})
        
        processed_dataset[split] = Dataset.from_dict(flat_data)
    
    dataset_dict = DatasetDict(processed_dataset)
    print(f"Loaded {len(dataset_dict['train'])} train, {len(dataset_dict['validation'])} val examples")
    return dataset_dict


def prepare_datasets(dataset: DatasetDict, tokenizer, max_length=384, doc_stride=128, num_proc=1):
    """Prepare datasets with tokenization."""
    preprocessor = QADataPreprocessor(tokenizer, max_length, doc_stride, tokenizer.padding_side == "right")
    
    print("Tokenizing data...")
    tokenized_train = dataset["train"].map(
        preprocessor.prepare_train_features, batched=True, num_proc=num_proc,
        remove_columns=dataset["train"].column_names, desc="Train")
    
    tokenized_val = dataset["validation"].map(
        preprocessor.prepare_validation_features, batched=True, num_proc=num_proc,
        remove_columns=dataset["validation"].column_names, desc="Validation")
    
    return tokenized_train, tokenized_val, dataset["validation"]


def load_and_prepare_data(train_file, validation_file, model_name="microsoft/deberta-v3-base", 
                          max_length=384, doc_stride=128, num_proc=1):
    """Load and prepare data from local JSON files."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Ensure we have a fast tokenizer (required for offset_mapping)
    if not tokenizer.is_fast:
        print("Slow tokenizer detected, loading fast tokenizer explicitly...")
        if "deberta" in model_name.lower():
            from transformers import DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
        else:
            raise ValueError(f"Fast tokenizer not available for {model_name}")
    
    dataset = load_qa_dataset_from_files(train_file, validation_file)
    tokenized_train, tokenized_val, raw_val = prepare_datasets(dataset, tokenizer, max_length, doc_stride, num_proc)
    print(f"Ready: {len(tokenized_train)} train, {len(tokenized_val)} val examples")
    return tokenized_train, tokenized_val, raw_val, tokenizer

