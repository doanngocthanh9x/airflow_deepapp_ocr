"""
NER Inference Examples using PhoBERT
Ví dụ sử dụng model PhoBERT NER đã được huấn luyện
"""

import json
import logging
import os
import ssl
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.request import ssl as urllib_ssl

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

# Fix SSL certificate verification issue
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

# Disable SSL verification for requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class PhoBERTNER:
    """
    Vietnamese NER using PhoBERT
    Trích xuất Named Entities từ văn bản tiếng Việt
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize NER model
        
        Args:
            model_path: Path to trained model directory (default: models/phobert_ner)
        """
        if model_path is None:
            # Default to local downloaded model
            project_root = Path(__file__).parent.parent.parent
            model_path = str(project_root / "models" / "phobert_ner")
        
        self.model_path = model_path
        
        # Check if it's a local path (not a HuggingFace repo name with '/')
        is_hf_model = '/' in model_path and not Path(model_path).exists()
        
        if is_hf_model:
            # Load from HuggingFace with SSL verification disabled
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Get labels from model config
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
                
                logger.info(f"Loaded PhoBERT model from HuggingFace: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace: {e}")
                raise
        else:
            # Load from local path
            self.model_path = Path(model_path)
            
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model directory not found: {self.model_path}\n"
                    f"Please download the model first by running: python download_phobert_smart.py\n"
                    f"Or train a model by running the DAG: dag_ner_from_text"
                )
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=True,  # Use fast tokenizer for word_ids support
            )
            
            # Load model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                weights_only=False,
                dtype=torch.float32,
            )
            
            # Load label config
            config_path = self.model_path / 'label_config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.id2label = {int(k): v for k, v in config['id2label'].items()}
                self.label2id = {v: int(k) for k, v in config['id2label'].items()}
            else:
                # Use model's config if label_config.json doesn't exist
                self.id2label = self.model.config.id2label
                self.label2id = self.model.config.label2id
            
            logger.info(f"Loaded model from local path: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Device: {self.device}")
    
    def tokenize(self, text: str) -> Tuple[List[int], List[int]]:
        """
        Tokenize text and return input_ids and attention_mask
        
        Args:
            text: Input Vietnamese text (word-segmented)
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        tokens = text.split()
        
        encoded = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
        )
        
        return encoded
    
    def predict(self, text: str, return_confidence: bool = False) -> List[Dict]:
        """
        Predict NER tags for input text
        
        Args:
            text: Input Vietnamese text (word-segmented)
            return_confidence: If True, return confidence scores
            
        Returns:
            List of dicts with format:
            [
                {'word': 'Công', 'entity': 'B-ORG', 'confidence': 0.98},
                {'word': 'ty', 'entity': 'I-ORG', 'confidence': 0.97},
                ...
            ]
        """
        tokens = text.split()
        
        # Tokenize with offset_mapping
        encoded = self.tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Try to get word_ids, fallback to offset_mapping if not available
        try:
            word_ids = encoded.word_ids()[0]
        except ValueError:
            # Fallback: use offset_mapping
            logger.debug("word_ids not available, using offset_mapping")
            word_ids = None
        
        results = []
        prev_word_idx = None
        
        if word_ids is not None:
            # Use word_ids mapping
            for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
                if word_id is None:
                    continue
                
                if word_id != prev_word_idx:
                    label = self.id2label.get(int(pred), 'O')
                    confidence = float(torch.softmax(logits[idx], dim=-1).max())
                    
                    if word_id < len(tokens):
                        results.append({
                            'word': tokens[word_id],
                            'entity': label,
                            'confidence': confidence if return_confidence else None,
                        })
                
                prev_word_idx = word_id
        else:
            # Fallback: map tokens to subword tokens manually
            token_idx = 0
            for idx, (input_id, pred) in enumerate(zip(input_ids[0], predictions)):
                # Skip special tokens
                if input_id in self.tokenizer.all_special_ids:
                    continue
                
                if idx > 0:  # Skip first token (usually [CLS])
                    label = self.id2label.get(int(pred), 'O')
                    confidence = float(torch.softmax(logits[idx], dim=-1).max())
                    
                    # Map to word if this is first subword
                    if token_idx < len(tokens):
                        results.append({
                            'word': tokens[token_idx] if token_idx == 0 or label.startswith('B-') else '',
                            'entity': label,
                            'confidence': confidence if return_confidence else None,
                        })
                        if label.startswith('B-'):
                            token_idx += 1
        
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type
        
        Args:
            text: Input Vietnamese text (word-segmented)
            
        Returns:
            Dict mapping entity type to list of entity values
            {
                'PER': ['Nguyễn Văn A', 'Hồ Chí Minh'],
                'ORG': ['Google', 'Microsoft'],
                'LOC': ['Hà Nội', 'Hồ Chí Minh City'],
                ...
            }
        """
        predictions = self.predict(text)
        
        entities = {}
        current_entity = None
        current_type = None
        
        for item in predictions:
            token = item['word']
            tag = item['entity']
            
            if tag == 'O':
                if current_entity is not None:
                    entity_type = current_type.replace('B-', '').replace('I-', '')
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(' '.join(current_entity))
                    current_entity = None
                    current_type = None
            
            elif tag.startswith('B-'):
                # Start of new entity
                if current_entity is not None:
                    entity_type = current_type.replace('B-', '').replace('I-', '')
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(' '.join(current_entity))
                
                current_entity = [token]
                current_type = tag
            
            elif tag.startswith('I-'):
                # Continuation of entity
                if current_entity is not None:
                    current_entity.append(token)
        
        # Don't forget last entity
        if current_entity is not None:
            entity_type = current_type.replace('B-', '').replace('I-', '')
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(' '.join(current_entity))
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        Batch prediction for multiple texts
        
        Args:
            texts: List of input texts (word-segmented)
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]


# ============================================================================
# USAGE EXAMPLES - Các ví dụ sử dụng
# ============================================================================

def example_basic_usage():
    """
    Ví dụ 1: Cách sử dụng cơ bản
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Initialize model
    model_path = '/opt/airflow/models/phobert_ner/final_model'
    ner = PhoBERTNER(model_path)
    
    # Example Vietnamese text (MUST BE WORD-SEGMENTED)
    text = "Công ty Google tuyển dụng kỹ sư tại Hà Nội"
    
    # Predict
    results = ner.predict(text, return_confidence=True)
    
    print(f"Input: {text}\n")
    print("Token-level predictions:")
    for item in results:
        if item['confidence']:
            print(f"  {item['word']:15} → {item['entity']:10} ({item['confidence']:.3f})")
        else:
            print(f"  {item['word']:15} → {item['entity']:10}")


def example_entity_extraction():
    """
    Ví dụ 2: Trích xuất entities theo loại
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Entity Extraction")
    print("=" * 60)
    
    model_path = '/opt/airflow/models/phobert_ner/final_model'
    ner = PhoBERTNER(model_path)
    
    text = "Nguyễn Văn A làm việc tại công ty Microsoft ở Hà Nội từ năm 2020"
    
    entities = ner.extract_entities(text)
    
    print(f"Input: {text}\n")
    print("Extracted entities:")
    for entity_type, values in entities.items():
        print(f"  {entity_type}:")
        for value in values:
            print(f"    - {value}")


def example_batch_processing():
    """
    Ví dụ 3: Xử lý batch
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    model_path = '/opt/airflow/models/phobert_ner/final_model'
    ner = PhoBERTNER(model_path)
    
    texts = [
        "Công ty Apple ra mắt iPhone tại Mỹ",
        "Hôm qua tôi gặp Hồ Chí Minh tại Đà Nẵng",
        "Học viện Công nghệ Bưu chính Viễn thông tuyển sinh",
    ]
    
    results = ner.predict_batch(texts)
    
    print("Batch prediction results:\n")
    for text, preds in zip(texts, results):
        print(f"Input: {text}")
        entities_str = ' | '.join([f"{p['word']}/{p['entity']}" for p in preds])
        print(f"Tags: {entities_str}\n")


def example_pipeline_api():
    """
    Ví dụ 4: Dùng HuggingFace pipeline API (đơn giản nhất)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: HuggingFace Pipeline API")
    print("=" * 60)
    
    model_path = '/opt/airflow/models/phobert_ner/final_model'
    
    # Create NER pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="simple",
    )
    
    text = "Công ty Google tuyển dụng kỹ sư tại Hà Nội"
    
    # Predict
    results = ner_pipeline(text)
    
    print(f"Input: {text}\n")
    print("Results:")
    for result in results:
        print(f"  {result['word']:20} → {result['entity']:10} ({result['score']:.3f})")


def example_custom_formatting():
    """
    Ví dụ 5: Custom output formatting
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Output Formatting")
    print("=" * 60)
    
    model_path = '/opt/airflow/models/phobert_ner/final_model'
    ner = PhoBERTNER(model_path)
    
    texts = [
        "Nguyễn Văn A làm việc tại Google ở Hà Nội",
        "Apple ra mắt iPhone 15 vào tháng 9 năm 2023",
    ]
    
    print("Formatted output:\n")
    for text in texts:
        print(f"📝 Text: {text}")
        
        entities = ner.extract_entities(text)
        
        if entities:
            print("   🏷️ Entities:")
            for entity_type, values in entities.items():
                emoji_map = {
                    'PER': '👤',
                    'ORG': '🏢',
                    'LOC': '📍',
                    'DATE': '📅',
                    'MISC': '📌',
                }
                emoji = emoji_map.get(entity_type, '•')
                print(f"      {emoji} {entity_type}: {', '.join(values)}")
        else:
            print("   No entities found")
        
        print()


# ============================================================================
# Command line usage
# ============================================================================

if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python inference.py [model_path] '<text>'")
        print("  If model_path not provided, uses default: models/phobert_ner")
        print("\nExamples:")
        print("  python inference.py 'Công ty Google tuyển dụng kỹ sư'")
        print("  python inference.py /path/to/model 'Hồ Chí Minh sinh năm 1890'")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Only text provided, use default model
        model_path = None
        text = sys.argv[1]
    else:
        # model_path and text provided
        model_path = sys.argv[1]
        text = sys.argv[2]
    
    if not text:
        # Run all examples
        example_basic_usage()
        example_entity_extraction()
        example_batch_processing()
        example_pipeline_api()
        example_custom_formatting()
    else:
        # Process single text
        ner = PhoBERTNER(model_path)
        
        print(f"\nInput: {text}\n")
        
        # Token-level predictions
        print("Token-level predictions:")
        results = ner.predict(text, return_confidence=True)
        for item in results:
            conf_str = f"({item['confidence']:.3f})" if item['confidence'] else ""
            print(f"  {item['word']:20} → {item['entity']:10} {conf_str}")
        
        # Entity-level extraction
        print("\nExtracted entities:")
        entities = ner.extract_entities(text)
        if entities:
            for entity_type, values in entities.items():
                print(f"  {entity_type}: {', '.join(values)}")
        else:
            print("  No entities found")
