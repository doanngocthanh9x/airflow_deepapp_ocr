"""
Configuration for NER DAG using PhoBERT
Named Entity Recognition for Vietnamese text
"""
from pathlib import Path

# Paths
BASE_DIR = Path('/opt/airflow')
DATA_DIR = BASE_DIR / 'dags' / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Input/Output directories
INPUT_DIR = DATA_DIR / 'ner_text_input'    # Raw text files
OUTPUT_DIR = DATA_DIR / 'ner_text_output'  # Prediction results
DATASET_DIR = DATA_DIR / 'ner_dataset'     # Train/val/test splits
MODEL_DIR = MODELS_DIR / 'phobert_ner'

# NER Model Configuration
NER_CONFIG = {
    'model_name': 'vinai/phobert-base',     # PhoBERT base model
    'model_type': 'roberta',
    'max_seq_length': 256,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 5e-5,
    'warmup_steps': 500,
}

# Label set for Vietnamese NER
# B-* = Beginning of entity
# I-* = Inside entity
# O = Outside entity
NER_LABELS = [
    'O',        # Outside
    'B-PER',    # Begin Person
    'I-PER',    # Inside Person
    'B-ORG',    # Begin Organization
    'I-ORG',    # Inside Organization
    'B-LOC',    # Begin Location
    'I-LOC',    # Inside Location
    'B-DATE',   # Begin Date
    'I-DATE',   # Inside Date
    'B-MISC',   # Begin Miscellaneous
    'I-MISC',   # Inside Miscellaneous
]

# Data format for NER
# Each line = one token + space + label
# Empty line = sentence separator
DATA_FORMAT = {
    'token_column': 0,
    'label_column': 1,
    'separator': ' ',  # tab or space separated
}

# Training configuration
TRAIN_CONFIG = {
    'fp16': False,           # Mixed precision
    'do_train': True,
    'do_eval': True,
    'do_predict': True,
    'save_strategy': 'epoch',
    'eval_strategy': 'epoch',
    'logging_steps': 50,
    'save_total_limit': 3,
    'seed': 42,
}
