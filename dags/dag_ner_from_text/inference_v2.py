"""
PhoBERT NER Inference - Correct Implementation
Based on: https://github.com/Avi197/Phobert-Named-Entity-Reconigtion
"""

import torch
import logging
import sys
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Add parent paths to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import path manager
from pathUtlis import PathManager

logger = logging.getLogger(__name__)

# Try to use VnCoreNLP for word segmentation
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    logger.warning("underthesea not installed. Use simple word split instead.")


class PhoBERTNER:
    """Vietnamese NER using PhoBERT"""
    
    def __init__(self, model_path: str = None, model_size: str = "base"):
        """
        Initialize NER model
        
        Args:
            model_path: Path to trained model (default: auto-detect)
            model_size: Which model to use if model_path is None: "base" or "large"
        """
        if model_path is None:
            # Use PathManager to get project root
            project_root = PathManager.get_root()
            
            if model_size.lower() == "large":
                model_path = str(project_root / "models" / "phobert_ner_large")
                logger.info("Using PhoBERT large model")
            else:
                model_path = str(project_root / "models" / "phobert_ner")
                logger.info("Using PhoBERT base model")
        
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: python download_phobert_smart.py"
            )
        
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            use_fast=False,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            weights_only=False,
        )
        
        # Load label config
        import json
        config_path = self.model_path / 'label_config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.id2label = {int(k): v for k, v in config['id2label'].items()}
        else:
            self.id2label = self.model.config.id2label
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Device: {self.device}")
    
    def segment_text(self, text: str) -> List[str]:
        """
        Word segment Vietnamese text
        
        Args:
            text: Raw Vietnamese text
            
        Returns:
            Word-segmented sentence
        """
        if HAS_UNDERTHESEA:
            try:
                # Use underthesea for word segmentation (returns string with _ separators)
                segmented = word_tokenize(text, format="text")
                return segmented.split()
            except Exception as e:
                logger.warning(f"underthesea error: {e}, using simple split")
        
        # Fallback: simple split
        return text.split()
    
    def predict(self, text: str, return_confidence: bool = False) -> List[Dict]:
        """
        Predict NER tags for input text
        
        Args:
            text: Raw Vietnamese text (will be segmented)
            return_confidence: If True, return confidence scores
            
        Returns:
            List of token-label predictions
        """
        # Word segmentation
        tokens = self.segment_text(text)
        sequence = " ".join(tokens)  # Create space-separated sequence
        
        # Encode
        list_ids = self.tokenizer.encode(sequence)
        input_ids = torch.tensor([list_ids])
        
        # Get tokens from ids (for output)
        output_tokens = self.tokenizer.convert_ids_to_tokens(list_ids)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
        
        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Map predictions to tokens
        results = []
        for token, pred_id in zip(output_tokens, predictions):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            # Get label
            label = self.id2label.get(int(pred_id), 'O')
            
            # Get confidence
            confidence = None
            if return_confidence:
                confidence = float(torch.softmax(logits[len(results)], dim=-1).max().cpu().numpy())
            
            # Handle subword tokens (those with @@)
            if '@@' in token:
                if results:
                    results[-1]['word'] += token[:-2]
                else:
                    results.append({
                        'word': token[:-2],
                        'entity': label,
                        'confidence': confidence,
                    })
            else:
                results.append({
                    'word': token,
                    'entity': label,
                    'confidence': confidence,
                })
        
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities grouped by type
        
        Args:
            text: Raw Vietnamese text
            
        Returns:
            Dict mapping entity type to list of entity values
        """
        predictions = self.predict(text)
        
        entities = {}
        current_entity = None
        current_label = None
        
        for pred in predictions:
            label = pred['entity']
            word = pred['word']
            
            if label == 'O':
                # Outside entity
                if current_entity is not None:
                    entity_type = current_label.replace('B-', '').replace('I-', '')
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(current_entity)
                    current_entity = None
                    current_label = None
            
            elif label.startswith('B-'):
                # Beginning of entity
                if current_entity is not None:
                    entity_type = current_label.replace('B-', '').replace('I-', '')
                    if entity_type not in entities:
                        entities[entity_type] = []
                    entities[entity_type].append(current_entity)
                
                current_entity = word
                current_label = label
            
            elif label.startswith('I-'):
                # Inside entity
                if current_entity is not None:
                    current_entity += ' ' + word
        
        # Don't forget last entity
        if current_entity is not None:
            entity_type = current_label.replace('B-', '').replace('I-', '')
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(current_entity)
        
        return entities


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_v2.py '<text>' [base|large]")
        print("Example: python inference_v2.py 'Công ty Google tuyển dụng kỹ sư tại Hà Nội' large")
        sys.exit(1)
    
    text = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "large"
    
    # Initialize model
    ner = PhoBERTNER(model_size=model_size)
    
    print(f"\nInput: {text}\n")
    
    # Token-level predictions
    print("Token-level predictions:")
    results = ner.predict(text, return_confidence=True)
    for item in results:
        if item['confidence'] is not None:
            print(f"  {item['word']:20} → {item['entity']:10} ({item['confidence']:.3f})")
        else:
            print(f"  {item['word']:20} → {item['entity']:10}")
    
    # Entity-level extraction
    print("\nExtracted entities:")
    entities = ner.extract_entities(text)
    if entities:
        for entity_type, values in entities.items():
            print(f"  {entity_type}: {', '.join(values)}")
    else:
        print("  No entities found")
