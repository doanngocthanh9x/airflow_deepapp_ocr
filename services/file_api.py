"""
File Manager API - FastAPI endpoints for file management
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
import shutil
import tempfile

from file_manager import FileManager, FileStatus, FileType, get_file_manager

app = FastAPI(
    title="File Manager API",
    description="API quản lý file trong pipeline data processing",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize FileManager
fm = get_file_manager()


@app.get("/")
async def root():
    """API root"""
    return {
        "service": "File Manager API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload",
            "list": "GET /files",
            "status": "GET /files/{file_id}",
            "queue": "GET /queue",
            "stats": "GET /stats",
            "move": "POST /files/{file_id}/move",
            "process": "POST /files/{file_id}/process",
        }
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    task_type: str = Query("ocr", description="Task type: ocr, classification, detection, segmentation, other")
):
    """
    Upload file mới vào rawdata
    
    - **file**: File để upload
    - **task_type**: Loại task sẽ xử lý file này
    """
    try:
        # Validate task type
        try:
            task = FileType(task_type)
        except ValueError:
            raise HTTPException(400, f"Invalid task_type. Must be one of: {[t.value for t in FileType]}")
        
        # Save to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        # Upload via FileManager
        metadata = fm.upload(tmp_path, task_type=task, metadata={
            'original_filename': file.filename,
            'content_type': file.content_type
        })
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return JSONResponse(content={
            "success": True,
            "message": f"File uploaded successfully",
            "file_id": metadata['file_id'],
            "metadata": metadata
        })
        
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/files")
async def list_files(
    stage: Optional[str] = Query(None, description="Filter by stage: rawdata, process, crop, dataset, failed, archive"),
    status: Optional[str] = Query(None, description="Filter by status"),
    task_type: Optional[str] = Query(None, description="Filter by task type")
):
    """Liệt kê tất cả files với filter tùy chọn"""
    files = fm.list_files(stage=stage, status=status, task_type=task_type)
    return {
        "count": len(files),
        "files": files
    }


@app.get("/files/{file_id}")
async def get_file_status(file_id: str):
    """Lấy trạng thái của file"""
    metadata = fm.get_status(file_id)
    if not metadata:
        raise HTTPException(404, f"File not found: {file_id}")
    return metadata


@app.get("/files/{file_id}/download")
async def download_file(file_id: str):
    """Download file"""
    metadata = fm.get_status(file_id)
    if not metadata:
        raise HTTPException(404, f"File not found: {file_id}")
    
    file_path = Path(metadata['current_path'])
    if not file_path.exists():
        raise HTTPException(404, f"File not found on disk: {file_path}")
    
    return FileResponse(
        path=str(file_path),
        filename=metadata['original_name'],
        media_type='application/octet-stream'
    )


@app.post("/files/{file_id}/move")
async def move_file(
    file_id: str,
    stage: str = Query(..., description="Target stage: process, crop, dataset, failed, archive"),
    split: Optional[str] = Query(None, description="For dataset stage: train, val, test"),
    label: Optional[str] = Query(None, description="For dataset stage: label/category")
):
    """Di chuyển file sang stage khác"""
    try:
        metadata = fm.get_status(file_id)
        if not metadata:
            raise HTTPException(404, f"File not found: {file_id}")
        
        if stage == 'process':
            result = fm.move_to_process(file_id)
        elif stage == 'crop':
            result = fm.move_to_crop(file_id)
        elif stage == 'dataset':
            result = fm.move_to_dataset(file_id, split=split or 'train', label=label)
        elif stage == 'failed':
            result = fm.mark_failed(file_id, error="Manually marked as failed")
        elif stage == 'archive':
            result = fm.archive(file_id)
        else:
            raise HTTPException(400, f"Invalid stage: {stage}")
        
        return {
            "success": True,
            "message": f"File moved to {stage}",
            "metadata": result
        }
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/files/{file_id}/process")
async def start_processing(file_id: str):
    """Đánh dấu bắt đầu xử lý file"""
    try:
        result = fm.start_processing(file_id)
        return {
            "success": True,
            "message": "Processing started",
            "metadata": result
        }
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/queue")
async def get_queue(
    task_type: Optional[str] = Query(None, description="Filter by task type")
):
    """Lấy danh sách file đang chờ xử lý"""
    queue = fm.get_queue(task_type=task_type)
    return {
        "count": len(queue),
        "queue": queue
    }


@app.get("/stats")
async def get_stats():
    """Thống kê tổng quan"""
    return fm.get_stats()


@app.delete("/cleanup")
async def cleanup_old_files(
    days: int = Query(30, description="Delete files older than N days"),
    stage: str = Query("archive", description="Stage to cleanup")
):
    """Xóa file cũ"""
    deleted = fm.cleanup_old_files(days=days, stage=stage)
    return {
        "success": True,
        "deleted_count": deleted,
        "message": f"Deleted {deleted} files older than {days} days from {stage}"
    }


@app.get("/stages")
async def list_stages():
    """Liệt kê các stage và thư mục"""
    return {
        "stages": list(fm.dirs.keys()),
        "paths": {name: str(path) for name, path in fm.dirs.items()},
        "file_types": [t.value for t in FileType],
        "statuses": [s.value for s in FileStatus]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
