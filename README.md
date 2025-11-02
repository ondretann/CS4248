# Extractive Question Answering System
### CS4248 Project - Natural Language Processing

This project implements an extractive question answering system using fine-tuned BERT-style transformer models, specifically DeBERTa-v3-base. The system is trained to identify answer spans within given contexts for input questions.

## 📋 Project Overview

### Approach
We fine-tune advanced BERT-style transformer models (primarily DeBERTa-v3-base) for extractive question answering tasks. The model uses combined cross-entropy loss for both the start and end token positions of the answer span.

### Uniqueness
Our approach distinguishes itself through:
- **Deliberate model selection**: Using DeBERTa-v3-base, which represents an advancement over the original BERT architecture
- **Ensemble techniques**: Combining multiple models to mitigate individual model weaknesses
- **Comprehensive evaluation**: Extensive testing and error analysis

### Datasets
- **Primary**: SQuAD (Stanford Question Answering Dataset)
- **Additional** (if needed):
  - NewsQA
  - SQuAD 2.0
  - Natural Questions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+Brief description of the approach / method / algorithm to be used
We will be finetuning BERT-style transformer models to perform the extractive question answer such as DeBERTa-v3-base and using combined cross entropy loss for both the start and end indexes of the tokens for the answer span.

What is the uniqueness of your proposed method (if any)? How does it compare to prior methods?
The uniqueness of our method is from the deliberate selection of particular advanced models and the subsequent application of ensemble techniques to maybe mitigate the individual weaknesses of each model. Prior methods have often relied on the original BERT architecture, which, while revolutionary, has been surpassed by more recent developments.

Any additional and publicly available training data to be used? Do they require any manual annotation?
If the given data is not enough we plan to use one or more of the following: NewsQA, Squad 2.0, Natural Questions. They do not require manual annotation as we could use the datasets python library that provides data with annotation out of the box.

Any external (supporting) code to be used?
Hugging Face has a library of BERT-style transformers that we can use to start with. It also has libraries for data pre-processing, training the model for fine-tuning, and evaluation. We will also likely follow the structure of the sample code in their documentation about question answering, and make our own changes from there.


### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Data Exploration

Start by exploring the dataset using the Jupyter notebook:

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

The notebook provides:
- Dataset statistics and visualizations
- Tokenization examples
- Data preprocessing demonstrations
- Training and evaluation examples

### 2. Training

#### Quick Start (Default Settings)
```bash
python scripts/train.py
```

#### Custom Training
```bash
python scripts/train.py \
    --model_name microsoft/deberta-v3-base \
    --dataset_name squad \
    --output_dir ./models/deberta-v3-base-squad \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --fp16 \
    --max_length 384 \
    --doc_stride 128
```

#### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `microsoft/deberta-v3-base` | Pre-trained model to fine-tune |
| `--dataset_name` | `squad` | Dataset to use |
| `--output_dir` | `./models/deberta-v3-base-squad` | Directory for saving models |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--per_device_train_batch_size` | `8` | Training batch size per GPU |
| `--learning_rate` | `3e-5` | Learning rate |
| `--fp16` | `False` | Use mixed precision training |
| `--max_length` | `384` | Maximum sequence length |
| `--doc_stride` | `128` | Stride for sliding window |

For all arguments:
```bash
python scripts/train.py --help
```

### 3. Evaluation

During training, the model is automatically evaluated at specified intervals. To evaluate a saved model:

```python
from src.evaluation import evaluate_model
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from src.data_preprocessing import load_and_prepare_data

# Load model
model = AutoModelForQuestionAnswering.from_pretrained("./models/deberta-v3-base-squad")
tokenizer = AutoTokenizer.from_pretrained("./models/deberta-v3-base-squad")

# Load data
_, eval_dataset, raw_eval_dataset, _ = load_and_prepare_data(
    model_name="microsoft/deberta-v3-base",
    dataset_name="squad"
)

# Evaluate
metrics, predictions = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    raw_dataset=raw_eval_dataset,
    batch_size=16
)

print(f"Exact Match: {metrics['exact_match']:.2f}%")
print(f"F1 Score: {metrics['f1']:.2f}%")
```

### 4. Making Predictions

#### Single Prediction
```bash
python scripts/predict.py \
    --model_path ./models/deberta-v3-base-squad \
    --question "What is the capital of France?" \
    --context "Paris is the capital and most populous city of France."
```

#### Batch Prediction
Create a JSON file with questions:
```json
[
    {
        "id": "1",
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France."
    },
    {
        "id": "2",
        "question": "Who wrote Romeo and Juliet?",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare."
    }
]
```

Run batch prediction:
```bash
python scripts/predict.py \
    --model_path ./models/deberta-v3-base-squad \
    --input_file questions.json \
    --output_file predictions.json
```

## 📊 Evaluation Metrics

We use standard QA evaluation metrics:

- **Exact Match (EM)**: Percentage of predictions that match ground truth exactly
- **F1 Score**: Token-level F1 score between prediction and ground truth

These metrics are computed following the official SQuAD evaluation protocol.

## 🔧 Advanced Usage

### Using Different Models

The codebase supports any HuggingFace model with a question-answering head:

```bash
# RoBERTa
python scripts/train.py --model_name roberta-base

# BERT
python scripts/train.py --model_name bert-base-uncased

# ELECTRA
python scripts/train.py --model_name google/electra-base-discriminator
```

### Training on Additional Datasets

```bash
# SQuAD 2.0 (with unanswerable questions)
python scripts/train.py --dataset_name squad_v2

# Natural Questions (requires additional configuration)
# See notebooks/exploration.ipynb for examples
```

### Ensemble Methods

To create an ensemble:

1. Train multiple models:
```bash
python scripts/train.py --model_name microsoft/deberta-v3-base --output_dir ./models/deberta
python scripts/train.py --model_name roberta-base --output_dir ./models/roberta
python scripts/train.py --model_name bert-base-uncased --output_dir ./models/bert
```

2. Implement ensemble logic (see notebooks for examples)

### Hyperparameter Tuning

Key hyperparameters to experiment with:
- Learning rate: `[1e-5, 2e-5, 3e-5, 5e-5]`
- Batch size: `[8, 16, 32]`
- Epochs: `[2, 3, 4]`
- Max length: `[256, 384, 512]`
- Doc stride: `[64, 128, 192]`

## 📈 Expected Results

Typical performance on SQuAD validation set:

| Model | Exact Match | F1 Score |
|-------|-------------|----------|
| DeBERTa-v3-base | ~82-84% | ~88-90% |
| RoBERTa-base | ~80-82% | ~87-89% |
| BERT-base | ~78-80% | ~85-87% |

*Note: Results may vary based on hyperparameters and training settings*

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `--per_device_train_batch_size`
- Reduce `--max_length`
- Enable `--gradient_accumulation_steps`

### Slow Training
- Enable `--fp16` (requires CUDA)
- Increase `--per_device_train_batch_size` if memory allows
- Use fewer `--num_proc` for data preprocessing

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install specific packages
pip install transformers datasets torch
```

## 📚 References

- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [HuggingFace Question Answering Guide](https://huggingface.co/docs/transformers/tasks/question_answering)

## 🤝 Contributing

This is a course project for CS4248. For questions or issues, please contact the project team.

## 📄 License

This project is for educational purposes as part of CS4248 coursework.

## 🙏 Acknowledgments

- HuggingFace for providing pre-trained models and libraries
- Stanford NLP Group for the SQuAD dataset
- Course instructors and TAs for guidance

---

**Project Team**: CS4248 Group [Your Group Number]  
**Course**: CS4248 Natural Language Processing  
**Institution**: National University of Singapore

