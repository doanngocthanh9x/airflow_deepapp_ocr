"""
Tasks for Remove Background DAG
Using rembg library with various AI models
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dag_remove_background.config import (
    INPUT_DIR, OUTPUT_DIR, REMBG_CONFIG, BATCH_CONFIG, REMBG_HOME
)

logger = logging.getLogger(__name__)


def setup_environment(**context) -> Dict:
    """
    Setup environment and validate directories
    """
    logger.info("=" * 60)
    logger.info("TASK: Setup Environment")
    logger.info("=" * 60)
    
    # Set rembg home for model cache
    os.environ['U2NET_HOME'] = str(REMBG_HOME)
    
    # Create directories
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REMBG_HOME.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Model cache: {REMBG_HOME}")
    
    # Scan input images
    supported = BATCH_CONFIG['supported_formats']
    images = []
    for ext in supported:
        images.extend(INPUT_DIR.glob(f'*{ext}'))
        images.extend(INPUT_DIR.glob(f'*{ext.upper()}'))
    
    images = sorted(set(images))
    
    if not images:
        logger.warning(f"No images found in {INPUT_DIR}")
        logger.info(f"Supported formats: {supported}")
        return {
            'status': 'no_images',
            'input_count': 0,
            'images': [],
        }
    
    logger.info(f"Found {len(images)} images to process")
    
    result = {
        'status': 'ready',
        'input_count': len(images),
        'images': [str(p) for p in images],
        'config': REMBG_CONFIG,
    }
    
    # Push to XCom
    context['ti'].xcom_push(key='setup_result', value=result)
    
    return result


def remove_background_batch(**context) -> Dict:
    """
    Remove background from images using rembg
    """
    logger.info("=" * 60)
    logger.info("TASK: Remove Background")
    logger.info("=" * 60)
    
    # Get setup result
    ti = context['ti']
    setup_result = ti.xcom_pull(task_ids='setup_environment', key='setup_result')
    
    if not setup_result or setup_result.get('status') == 'no_images':
        logger.warning("No images to process")
        return {'status': 'skipped', 'processed': 0}
    
    images = [Path(p) for p in setup_result['images']]
    
    # Import rembg
    try:
        from rembg import remove, new_session
        from PIL import Image
        import io
    except ImportError as e:
        logger.error(f"Failed to import rembg: {e}")
        logger.error("Install with: pip install rembg[gpu] onnxruntime-gpu")
        raise
    
    # Create session with specified model
    model_name = REMBG_CONFIG.get('model', 'u2net')
    logger.info(f"Loading model: {model_name}")
    session = new_session(model_name)
    
    # Process images
    results = []
    success_count = 0
    error_count = 0
    
    for i, img_path in enumerate(images, 1):
        try:
            logger.info(f"[{i}/{len(images)}] Processing: {img_path.name}")
            
            # Read image
            with open(img_path, 'rb') as f:
                input_data = f.read()
            
            # Remove background
            output_data = remove(
                input_data,
                session=session,
                alpha_matting=REMBG_CONFIG.get('alpha_matting', False),
                alpha_matting_foreground_threshold=REMBG_CONFIG.get('alpha_matting_fg_threshold', 240),
                alpha_matting_background_threshold=REMBG_CONFIG.get('alpha_matting_bg_threshold', 10),
                alpha_matting_erode_size=REMBG_CONFIG.get('alpha_matting_erode_size', 10),
                bgcolor=REMBG_CONFIG.get('bgcolor'),
            )
            
            # Save output
            output_format = REMBG_CONFIG.get('output_format', 'png')
            output_name = img_path.stem + f'.{output_format}'
            output_path = OUTPUT_DIR / output_name
            
            # Convert to PIL and save
            img_output = Image.open(io.BytesIO(output_data))
            img_output.save(output_path)
            
            results.append({
                'input': str(img_path),
                'output': str(output_path),
                'status': 'success',
            })
            success_count += 1
            logger.info(f"  ✓ Saved: {output_path.name}")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {img_path.name}: {e}")
            results.append({
                'input': str(img_path),
                'output': None,
                'status': 'error',
                'error': str(e),
            })
            error_count += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {success_count} success, {error_count} errors")
    logger.info("=" * 60)
    
    result = {
        'status': 'completed',
        'total': len(images),
        'success': success_count,
        'errors': error_count,
        'results': results,
    }
    
    ti.xcom_push(key='process_result', value=result)
    
    return result


def generate_report(**context) -> Dict:
    """
    Generate processing report
    """
    logger.info("=" * 60)
    logger.info("TASK: Generate Report")
    logger.info("=" * 60)
    
    ti = context['ti']
    process_result = ti.xcom_pull(task_ids='remove_background', key='process_result')
    
    if not process_result:
        logger.warning("No processing result found")
        return {'status': 'no_data'}
    
    # Create report
    report = {
        'generated_at': datetime.now().isoformat(),
        'config': REMBG_CONFIG,
        'summary': {
            'total_images': process_result.get('total', 0),
            'success': process_result.get('success', 0),
            'errors': process_result.get('errors', 0),
        },
        'results': process_result.get('results', []),
    }
    
    # Save report
    report_path = OUTPUT_DIR / 'report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Report saved: {report_path}")
    
    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 40)
    logger.info(f"Total: {report['summary']['total_images']}")
    logger.info(f"Success: {report['summary']['success']}")
    logger.info(f"Errors: {report['summary']['errors']}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    return report
