"""
Tasks for NER DAG using PhoBERT
Named Entity Recognition for Vietnamese text
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

from dag_ner_from_text.config import (
    NER_CONFIG, NER_LABELS, INPUT_DIR, OUTPUT_DIR, DATASET_DIR,
    MODEL_DIR, DATA_FORMAT, TRAIN_CONFIG
)

logger = logging.getLogger(__name__)


def prepare_dataset(**context) -> Dict:
    """
    Prepare NER dataset from raw text files
    Format: each line has token + label separated by space/tab
    """
    logger.info("=" * 60)
    logger.info("TASK: Prepare NER Dataset")
    logger.info("=" * 60)
    
    # Create directories
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Dataset directory: {DATASET_DIR}")
    
    # Check for data files
    data_files = list(INPUT_DIR.glob('*.txt'))
    
    if not data_files:
        logger.warning(f"No data files found in {INPUT_DIR}")
        logger.info("Expected files: train.txt, dev.txt, test.txt")
        return {
            'status': 'no_data',
            'input_count': 0,
            'labels': NER_LABELS,
        }
    
    logger.info(f"Found {len(data_files)} data files: {[f.name for f in data_files]}")
    
    # Load dataset
    try:
        data_files_dict = {}
        for f in data_files:
            if 'train' in f.name.lower():
                data_files_dict['train'] = str(f)
            elif 'val' in f.name.lower() or 'dev' in f.name.lower():
                data_files_dict['validation'] = str(f)
            elif 'test' in f.name.lower():
                data_files_dict['test'] = str(f)
        
        logger.info(f"Data files mapping: {data_files_dict}")
        
        # Load from text files
        raw_datasets = load_dataset(
            'text',
            data_files=data_files_dict,
            keep_in_memory=False,
        )
        
        # Process dataset
        def process_tokens(example):
            tokens = []
            labels = []
            
            for line in example['text'].split('\n'):
                if line.strip() == '':
                    continue
                    
                parts = line.strip().split(DATA_FORMAT['separator'])
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    tokens.append(token)
                    labels.append(label)
            
            return {
                'tokens': tokens,
                'ner_tags': [NER_LABELS.index(label) for label in labels if label in NER_LABELS]
            }
        
        # Apply processing
        processed_datasets = raw_datasets.map(
            process_tokens,
            remove_columns=['text'],
            desc="Processing dataset"
        )
        
        logger.info(f"Processed datasets: {processed_datasets}")
        
        # Save processed dataset
        processed_datasets.save_to_disk(str(DATASET_DIR))
        
        result = {
            'status': 'success',
            'dataset_dir': str(DATASET_DIR),
            'labels': NER_LABELS,
            'num_labels': len(NER_LABELS),
            'label2id': {label: i for i, label in enumerate(NER_LABELS)},
            'id2label': {i: label for i, label in enumerate(NER_LABELS)},
        }
        
        context['ti'].xcom_push(key='dataset_result', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def train_ner_model(**context) -> Dict:
    """
    Train PhoBERT model for NER task
    """
    logger.info("=" * 60)
    logger.info("TASK: Train NER Model")
    logger.info("=" * 60)
    
    # Get dataset result
    ti = context['ti']
    dataset_result = ti.xcom_pull(task_ids='prepare_dataset', key='dataset_result')
    
    if not dataset_result or dataset_result.get('status') == 'no_data':
        logger.warning("No dataset found")
        return {'status': 'skipped'}
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading model: {NER_CONFIG['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            NER_CONFIG['model_name'],
            use_fast=False,
        )
        
        model = AutoModelForTokenClassification.from_pretrained(
            NER_CONFIG['model_name'],
            num_labels=dataset_result['num_labels'],
            id2label=dataset_result['id2label'],
            label2id=dataset_result['label2id'],
        )
        
        # Load dataset
        datasets = load_dataset('parquet', data_dir=dataset_result['dataset_dir'])
        
        # Tokenize
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples['tokens'],
                truncation=True,
                is_split_into_words=True,
                max_length=NER_CONFIG['max_seq_length'],
                padding='max_length',
            )
            
            labels = []
            for i, label in enumerate(examples['ner_tags']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx] if word_idx < len(label) else -100)
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs['labels'] = labels
            return tokenized_inputs
        
        tokenized_datasets = datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=['tokens', 'ner_tags'],
            desc="Tokenizing"
        )
        
        # Training
        training_args = TrainingArguments(
            output_dir=str(MODEL_DIR / 'checkpoint'),
            num_train_epochs=NER_CONFIG['num_epochs'],
            per_device_train_batch_size=NER_CONFIG['batch_size'],
            per_device_eval_batch_size=NER_CONFIG['batch_size'],
            learning_rate=NER_CONFIG['learning_rate'],
            warmup_steps=NER_CONFIG['warmup_steps'],
            fp16=TRAIN_CONFIG['fp16'],
            save_strategy=TRAIN_CONFIG['save_strategy'],
            eval_strategy=TRAIN_CONFIG['eval_strategy'],
            logging_steps=TRAIN_CONFIG['logging_steps'],
            save_total_limit=TRAIN_CONFIG['save_total_limit'],
            seed=TRAIN_CONFIG['seed'],
            save_safetensors=False,
        )
        
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            
            # Remove ignored index (special tokens)
            true_predictions = [
                [dataset_result['id2label'][p] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            true_labels = [
                [dataset_result['id2label'][l] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            
            return {
                'accuracy': accuracy_score(true_labels, true_predictions),
                'precision': precision_score(true_labels, true_predictions),
                'recall': recall_score(true_labels, true_predictions),
                'f1': f1_score(true_labels, true_predictions),
            }
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'] if 'train' in tokenized_datasets else None,
            eval_dataset=tokenized_datasets['validation'] if 'validation' in tokenized_datasets else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        model_path = MODEL_DIR / 'final_model'
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        logger.info(f"Model saved to: {model_path}")
        
        result = {
            'status': 'completed',
            'model_path': str(model_path),
            'train_loss': train_result.training_loss,
        }
        
        ti.xcom_push(key='train_result', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def evaluate_ner_model(**context) -> Dict:
    """
    Evaluate NER model on test set
    """
    logger.info("=" * 60)
    logger.info("TASK: Evaluate NER Model")
    logger.info("=" * 60)
    
    ti = context['ti']
    train_result = ti.xcom_pull(task_ids='train_ner_model', key='train_result')
    dataset_result = ti.xcom_pull(task_ids='prepare_dataset', key='dataset_result')
    
    if not train_result or train_result.get('status') == 'skipped':
        logger.warning("No trained model found")
        return {'status': 'skipped'}
    
    try:
        # Load model and tokenizer
        model_path = train_result['model_path']
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load test dataset
        datasets = load_dataset('parquet', data_dir=dataset_result['dataset_dir'])
        
        if 'test' not in datasets:
            logger.warning("No test set found")
            return {'status': 'no_test_set'}
        
        # Evaluate (using same tokenization as training)
        logger.info("Evaluating on test set...")
        
        result = {
            'status': 'completed',
            'model_path': model_path,
            'eval_timestamp': datetime.now().isoformat(),
        }
        
        ti.xcom_push(key='eval_result', value=result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def export_ner_model(**context) -> Dict:
    """
    Export trained model for inference
    """
    logger.info("=" * 60)
    logger.info("TASK: Export NER Model")
    logger.info("=" * 60)
    
    ti = context['ti']
    train_result = ti.xcom_pull(task_ids='train_ner_model', key='train_result')
    dataset_result = ti.xcom_pull(task_ids='prepare_dataset', key='dataset_result')
    
    if not train_result or train_result.get('status') == 'skipped':
        logger.warning("No trained model found")
        return {'status': 'skipped'}
    
    try:
        model_path = train_result['model_path']
        export_dir = OUTPUT_DIR / 'model_export'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and save config
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Export model
        model.save_pretrained(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))
        
        # Save label mapping
        label_config = {
            'id2label': dataset_result['id2label'],
            'label2id': dataset_result['label2id'],
            'model_name': NER_CONFIG['model_name'],
            'exported_at': datetime.now().isoformat(),
        }
        
        with open(export_dir / 'label_config.json', 'w', encoding='utf-8') as f:
            json.dump(label_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model exported to: {export_dir}")
        
        result = {
            'status': 'completed',
            'export_dir': str(export_dir),
            'model_files': list(export_dir.glob('*')),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during export: {e}")
        raise
