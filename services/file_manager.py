"""
File Manager Service
Quản lý file qua các giai đoạn processing pipeline:
    rawdata -> process -> crop -> dataset

Author: Auto-generated
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileStatus(str, Enum):
    """Trạng thái file trong pipeline"""
    UPLOADED = "uploaded"           # Vừa upload, chưa xử lý
    QUEUED = "queued"              # Đang chờ trong queue
    PROCESSING = "processing"       # Đang được xử lý
    PROCESSED = "processed"         # Đã xử lý xong
    CROPPED = "cropped"            # Đã crop xong
    READY = "ready"                # Sẵn sàng cho training
    FAILED = "failed"              # Xử lý thất bại
    ARCHIVED = "archived"          # Đã archive


class FileType(str, Enum):
    """Loại file/task"""
    OCR = "ocr"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    OTHER = "other"


class FileManager:
    """
    Quản lý file trong pipeline data processing
    
    Cấu trúc thư mục:
        data/
        ├── rawdata/          # File upload mới
        ├── process/          # Đang xử lý
        │   ├── ocr/
        │   ├── classification/
        │   └── ...
        ├── crop/             # Kết quả crop
        ├── dataset/          # Dataset cho training
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── failed/           # File xử lý thất bại
        ├── archive/          # File đã archive
        └── metadata/         # Metadata JSON files
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize FileManager
        
        Args:
            base_path: Đường dẫn gốc (mặc định: /opt/airflow/data hoặc C:/Automation/Airflow/data)
        """
        if base_path is None:
            # Auto-detect environment
            if os.path.exists('/opt/airflow'):
                base_path = '/opt/airflow/data'
            else:
                base_path = 'C:/Automation/Airflow/data'
        
        self.base_path = Path(base_path)
        self._init_directories()
        
    def _init_directories(self):
        """Tạo cấu trúc thư mục nếu chưa có"""
        self.dirs = {
            'rawdata': self.base_path / 'rawdata',
            'process': self.base_path / 'process',
            'crop': self.base_path / 'crop',
            'dataset': self.base_path / 'dataset',
            'failed': self.base_path / 'failed',
            'archive': self.base_path / 'archive',
            'metadata': self.base_path / 'metadata',
        }
        
        # Create all directories
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories for process
        for task_type in FileType:
            (self.dirs['process'] / task_type.value).mkdir(exist_ok=True)
            
        # Create subdirectories for dataset
        for split in ['train', 'val', 'test']:
            (self.dirs['dataset'] / split).mkdir(exist_ok=True)
            
        logger.info(f"FileManager initialized at: {self.base_path}")
        
    def _get_file_hash(self, file_path: Path) -> str:
        """Tính MD5 hash của file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_file_id(self, filename: str) -> str:
        """Tạo unique file ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = Path(filename).stem
        return f"{timestamp}_{name}"
    
    def _get_metadata_path(self, file_id: str) -> Path:
        """Lấy đường dẫn metadata file"""
        return self.dirs['metadata'] / f"{file_id}.json"
    
    def _load_metadata(self, file_id: str) -> Optional[Dict]:
        """Load metadata của file"""
        meta_path = self._get_metadata_path(file_id)
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _save_metadata(self, file_id: str, metadata: Dict):
        """Lưu metadata của file"""
        meta_path = self._get_metadata_path(file_id)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    def upload(self, source_path: str, task_type: FileType = FileType.OCR, 
               metadata: Dict = None) -> Dict:
        """
        Upload file vào rawdata
        
        Args:
            source_path: Đường dẫn file nguồn
            task_type: Loại task sẽ xử lý
            metadata: Metadata bổ sung
            
        Returns:
            Dict với thông tin file đã upload
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        # Generate file ID and destination
        file_id = self._generate_file_id(source.name)
        dest_name = f"{file_id}{source.suffix}"
        dest_path = self.dirs['rawdata'] / dest_name
        
        # Copy file
        shutil.copy2(source, dest_path)
        
        # Create metadata
        file_metadata = {
            'file_id': file_id,
            'original_name': source.name,
            'current_name': dest_name,
            'current_path': str(dest_path),
            'current_stage': 'rawdata',
            'status': FileStatus.UPLOADED.value,
            'task_type': task_type.value,
            'file_hash': self._get_file_hash(dest_path),
            'file_size': dest_path.stat().st_size,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'history': [
                {
                    'action': 'upload',
                    'stage': 'rawdata',
                    'timestamp': datetime.now().isoformat(),
                    'details': {'source': str(source)}
                }
            ],
            'custom_metadata': metadata or {}
        }
        
        self._save_metadata(file_id, file_metadata)
        logger.info(f"Uploaded: {source.name} -> {dest_path} (ID: {file_id})")
        
        return file_metadata
    
    def move_to_process(self, file_id: str) -> Dict:
        """Di chuyển file từ rawdata sang process"""
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        if metadata['current_stage'] != 'rawdata':
            raise ValueError(f"File not in rawdata stage: {metadata['current_stage']}")
        
        # Determine destination based on task type
        task_type = metadata['task_type']
        source = Path(metadata['current_path'])
        dest = self.dirs['process'] / task_type / source.name
        
        # Move file
        shutil.move(str(source), str(dest))
        
        # Update metadata
        metadata['current_path'] = str(dest)
        metadata['current_stage'] = 'process'
        metadata['status'] = FileStatus.QUEUED.value
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['history'].append({
            'action': 'move',
            'stage': 'process',
            'timestamp': datetime.now().isoformat(),
            'details': {'task_type': task_type}
        })
        
        self._save_metadata(file_id, metadata)
        logger.info(f"Moved to process: {file_id} -> {dest}")
        
        return metadata
    
    def start_processing(self, file_id: str) -> Dict:
        """Đánh dấu file đang được xử lý"""
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        metadata['status'] = FileStatus.PROCESSING.value
        metadata['processing_started_at'] = datetime.now().isoformat()
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['history'].append({
            'action': 'start_processing',
            'stage': metadata['current_stage'],
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_metadata(file_id, metadata)
        return metadata
    
    def move_to_crop(self, file_id: str, crop_files: List[str] = None) -> Dict:
        """
        Di chuyển file sang crop stage
        
        Args:
            file_id: ID của file gốc
            crop_files: Danh sách các file crop đã tạo
        """
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        source = Path(metadata['current_path'])
        dest = self.dirs['crop'] / source.name
        
        # Move main file
        if source.exists():
            shutil.move(str(source), str(dest))
        
        # Update metadata
        metadata['current_path'] = str(dest)
        metadata['current_stage'] = 'crop'
        metadata['status'] = FileStatus.CROPPED.value
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['crop_files'] = crop_files or []
        metadata['history'].append({
            'action': 'move',
            'stage': 'crop',
            'timestamp': datetime.now().isoformat(),
            'details': {'crop_count': len(crop_files) if crop_files else 0}
        })
        
        self._save_metadata(file_id, metadata)
        logger.info(f"Moved to crop: {file_id} -> {dest}")
        
        return metadata
    
    def move_to_dataset(self, file_id: str, split: str = 'train', 
                        label: str = None) -> Dict:
        """
        Di chuyển file sang dataset cho training
        
        Args:
            file_id: ID của file
            split: train/val/test
            label: Label của file (optional)
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}")
        
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        source = Path(metadata['current_path'])
        
        # Create label subdirectory if needed
        if label:
            dest_dir = self.dirs['dataset'] / split / label
            dest_dir.mkdir(exist_ok=True)
        else:
            dest_dir = self.dirs['dataset'] / split
            
        dest = dest_dir / source.name
        
        # Move file
        if source.exists():
            shutil.move(str(source), str(dest))
        
        # Update metadata
        metadata['current_path'] = str(dest)
        metadata['current_stage'] = 'dataset'
        metadata['status'] = FileStatus.READY.value
        metadata['dataset_split'] = split
        metadata['label'] = label
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['history'].append({
            'action': 'move',
            'stage': 'dataset',
            'timestamp': datetime.now().isoformat(),
            'details': {'split': split, 'label': label}
        })
        
        self._save_metadata(file_id, metadata)
        logger.info(f"Moved to dataset/{split}: {file_id}")
        
        return metadata
    
    def mark_failed(self, file_id: str, error: str = None) -> Dict:
        """Đánh dấu file xử lý thất bại"""
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        source = Path(metadata['current_path'])
        dest = self.dirs['failed'] / source.name
        
        # Move file
        if source.exists():
            shutil.move(str(source), str(dest))
        
        # Update metadata
        metadata['current_path'] = str(dest)
        metadata['current_stage'] = 'failed'
        metadata['status'] = FileStatus.FAILED.value
        metadata['error'] = error
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['history'].append({
            'action': 'failed',
            'stage': 'failed',
            'timestamp': datetime.now().isoformat(),
            'details': {'error': error}
        })
        
        self._save_metadata(file_id, metadata)
        logger.warning(f"Marked as failed: {file_id} - {error}")
        
        return metadata
    
    def archive(self, file_id: str) -> Dict:
        """Archive file đã xử lý xong"""
        metadata = self._load_metadata(file_id)
        if not metadata:
            raise ValueError(f"File not found: {file_id}")
        
        source = Path(metadata['current_path'])
        
        # Create archive subdirectory by date
        archive_date = datetime.now().strftime("%Y-%m-%d")
        archive_dir = self.dirs['archive'] / archive_date
        archive_dir.mkdir(exist_ok=True)
        
        dest = archive_dir / source.name
        
        # Move file
        if source.exists():
            shutil.move(str(source), str(dest))
        
        # Update metadata
        metadata['current_path'] = str(dest)
        metadata['current_stage'] = 'archive'
        metadata['status'] = FileStatus.ARCHIVED.value
        metadata['updated_at'] = datetime.now().isoformat()
        metadata['history'].append({
            'action': 'archive',
            'stage': 'archive',
            'timestamp': datetime.now().isoformat()
        })
        
        self._save_metadata(file_id, metadata)
        logger.info(f"Archived: {file_id} -> {dest}")
        
        return metadata
    
    def get_status(self, file_id: str) -> Optional[Dict]:
        """Lấy trạng thái hiện tại của file"""
        return self._load_metadata(file_id)
    
    def list_files(self, stage: str = None, status: str = None, 
                   task_type: str = None) -> List[Dict]:
        """
        Liệt kê files theo filter
        
        Args:
            stage: Filter theo stage (rawdata, process, crop, dataset, failed, archive)
            status: Filter theo status
            task_type: Filter theo task type
        """
        files = []
        
        for meta_file in self.dirs['metadata'].glob("*.json"):
            metadata = self._load_metadata(meta_file.stem)
            if metadata:
                # Apply filters
                if stage and metadata.get('current_stage') != stage:
                    continue
                if status and metadata.get('status') != status:
                    continue
                if task_type and metadata.get('task_type') != task_type:
                    continue
                files.append(metadata)
        
        # Sort by updated_at descending
        files.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        return files
    
    def get_queue(self, task_type: str = None) -> List[Dict]:
        """Lấy danh sách file đang chờ xử lý"""
        return self.list_files(stage='process', status=FileStatus.QUEUED.value, 
                               task_type=task_type)
    
    def get_stats(self) -> Dict:
        """Thống kê tổng quan"""
        all_files = self.list_files()
        
        stats = {
            'total_files': len(all_files),
            'by_stage': {},
            'by_status': {},
            'by_task_type': {},
            'storage_size': {}
        }
        
        for f in all_files:
            # Count by stage
            stage = f.get('current_stage', 'unknown')
            stats['by_stage'][stage] = stats['by_stage'].get(stage, 0) + 1
            
            # Count by status
            status = f.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Count by task type
            task = f.get('task_type', 'unknown')
            stats['by_task_type'][task] = stats['by_task_type'].get(task, 0) + 1
        
        # Calculate storage size
        for name, path in self.dirs.items():
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            stats['storage_size'][name] = {
                'bytes': total_size,
                'mb': round(total_size / (1024 * 1024), 2)
            }
        
        return stats
    
    def cleanup_old_files(self, days: int = 30, stage: str = 'archive') -> int:
        """Xóa file cũ hơn N ngày trong stage"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        files = self.list_files(stage=stage)
        for f in files:
            updated_at = datetime.fromisoformat(f.get('updated_at', datetime.now().isoformat()))
            if updated_at < cutoff_date:
                file_path = Path(f['current_path'])
                meta_path = self._get_metadata_path(f['file_id'])
                
                if file_path.exists():
                    file_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                    
                deleted_count += 1
                logger.info(f"Deleted old file: {f['file_id']}")
        
        return deleted_count


# Singleton instance
_file_manager = None

def get_file_manager(base_path: str = None) -> FileManager:
    """Get singleton FileManager instance"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager(base_path)
    return _file_manager


if __name__ == "__main__":
    # Test FileManager
    fm = FileManager()
    print(f"Base path: {fm.base_path}")
    print(f"Directories: {list(fm.dirs.keys())}")
    print(f"Stats: {json.dumps(fm.get_stats(), indent=2)}")
