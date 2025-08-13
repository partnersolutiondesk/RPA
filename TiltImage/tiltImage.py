import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import logging

try:
    from PIL import Image, ImageEnhance
    import pdf2image
    import pytesseract
    import cv2
    import numpy as np
except ImportError as e:
    raise ImportError(f"Missing required library: {e}. Install with: pip install pillow pdf2image pytesseract opencv-python numpy")

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file in output directory"""
    log_file = output_dir / "processing.log"

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Create logger
    logger = logging.getLogger('document_rotator')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()
    logger.addHandler(file_handler)

    return logger

def detect_orientation_tesseract(image: Image.Image, logger: logging.Logger) -> int:
    """
    Detect orientation using Tesseract OCR
    Returns rotation angle needed to correct orientation (0, 90, 180, 270)
    """
    try:
        # Convert PIL image to numpy array for OpenCV
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Use Tesseract to detect orientation
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)

        # Get detected orientation
        orientation = osd.get('orientation', 0)
        confidence = osd.get('orientation_conf', 0)

        logger.info(f"Detected orientation: {orientation}° (confidence: {confidence})")

        # Convert orientation to rotation angle needed for correction
        rotation_map = {0: 0, 90: 270, 180: 180, 270: 90}
        return rotation_map.get(orientation, 0)

    except Exception as e:
        logger.warning(f"Tesseract orientation detection failed: {e}")
        return detect_orientation_fallback(image, logger)

def detect_orientation_fallback(image: Image.Image, logger: logging.Logger) -> int:
    """
    Fallback method: Try all orientations and pick best OCR confidence
    """
    logger.info("Using fallback orientation detection...")

    best_angle = 0
    best_confidence = 0

    for angle in [0, 90, 180, 270]:
        try:
            # Rotate image
            test_image = image.rotate(-angle, expand=True)

            # Try OCR and get confidence
            text = pytesseract.image_to_string(test_image)
            confidence = len([word for word in text.split() if word.isalpha()])

            logger.info(f"Angle {angle}°: confidence score = {confidence}")

            if confidence > best_confidence:
                best_confidence = confidence
                best_angle = angle

        except Exception as e:
            logger.warning(f"Failed to test angle {angle}°: {e}")
            continue

    logger.info(f"Best orientation: {best_angle}° rotation needed")
    return best_angle

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better OCR results
    """
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    return image

def rotate_image(image: Image.Image, angle: int, logger: logging.Logger) -> Image.Image:
    """
    Rotate image by specified angle
    """
    if angle == 0:
        return image

    # Rotate counter-clockwise (PIL convention)
    rotated = image.rotate(-angle, expand=True, fillcolor='white')
    logger.info(f"Rotated image by {angle}°")
    return rotated

def process_pdf(pdf_path: Path, output_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Process PDF file: convert to images, detect orientation, and correct
    """
    logger.info(f"Processing PDF: {pdf_path}")

    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=300,  # High DPI for better OCR
            fmt='PNG'
        )

        output_paths = []

        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")

            # Enhance image for OCR
            enhanced_image = enhance_image_for_ocr(image.copy())

            # Detect orientation
            rotation_angle = detect_orientation_tesseract(enhanced_image, logger)

            # Rotate original image (not enhanced version)
            corrected_image = rotate_image(image, rotation_angle, logger)

            # Save corrected image
            output_filename = f"{pdf_path.stem}_page_{i+1:03d}_corrected.png"
            output_path = output_dir / output_filename
            corrected_image.save(output_path, 'PNG', quality=95, dpi=(300, 300))
            output_paths.append(output_path)

            logger.info(f"Saved corrected page: {output_path}")

        return output_paths

    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        return []

def process_image(image_path: Path, output_dir: Path, logger: logging.Logger) -> Path:
    """
    Process image file: detect orientation and correct
    """
    logger.info(f"Processing image: {image_path}")

    try:
        # Load image
        image = Image.open(image_path)

        # Enhance image for OCR
        enhanced_image = enhance_image_for_ocr(image.copy())

        # Detect orientation
        rotation_angle = detect_orientation_tesseract(enhanced_image, logger)

        # Rotate original image
        corrected_image = rotate_image(image, rotation_angle, logger)

        # Save corrected image
        output_filename = f"{image_path.stem}_corrected{image_path.suffix}"
        output_path = output_dir / output_filename
        corrected_image.save(output_path, quality=95, dpi=(300, 300))

        logger.info(f"Saved corrected image: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        return None

def process_file(file_path: Path, output_dir: Path, logger: logging.Logger,
                 image_extensions: set, pdf_extensions: set) -> List[Path]:
    """
    Process a single file (image or PDF)
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    file_ext = file_path.suffix.lower()

    if file_ext in image_extensions:
        result = process_image(file_path, output_dir, logger)
        return [result] if result else []

    elif file_ext in pdf_extensions:
        return process_pdf(file_path, output_dir, logger)

    else:
        logger.error(f"Unsupported file type: {file_ext}")
        return []

def process_directory(directory_path: Path, output_dir: Path, logger: logging.Logger,
                      image_extensions: set, pdf_extensions: set) -> List[Path]:
    """
    Process all supported files in a directory
    """
    if not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return []

    all_outputs = []
    supported_extensions = image_extensions | pdf_extensions

    # Find all supported files (case-insensitive, no duplicates)
    supported_files = set()  # Use set to avoid duplicates

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            supported_files.add(file_path)

    supported_files = list(supported_files)  # Convert back to list
    logger.info(f"Found {len(supported_files)} supported files")

    for file_path in supported_files:
        outputs = process_file(file_path, output_dir, logger, image_extensions, pdf_extensions)
        all_outputs.extend(outputs)

    return all_outputs

def rotate_documents(config: Dict[str, Any]) -> List[Path]:
    """
    Main function to rotate documents based on configuration dictionary

    Args:
        config (dict): Configuration dictionary with the following keys:
            - 'input_path' (str): Path to input file or directory
            - 'output_dir' (str, optional): Output directory (default: 'rotated_documents')
            - 'image_extensions' (set, optional): Set of supported image extensions
            - 'pdf_extensions' (set, optional): Set of supported PDF extensions

    Returns:
        List[Path]: List of output file paths
    """

    # Default configuration
    default_config = {
        'output_dir': 'rotated_documents',
        'image_extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'},
        'pdf_extensions': {'.pdf'}
    }

    # Merge with user config
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # Validate required parameters
    if 'input_path' not in config:
        raise ValueError("'input_path' is required in configuration dictionary")

    # Setup paths
    input_path = Path(config['input_path'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=== Document Rotation Process Started ===")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output directory: {output_dir}")

    # Get extensions
    image_extensions = config['image_extensions']
    pdf_extensions = config['pdf_extensions']

    # Process input
    try:
        if input_path.is_file():
            outputs = process_file(input_path, output_dir, logger, image_extensions, pdf_extensions)
        elif input_path.is_dir():
            outputs = process_directory(input_path, output_dir, logger, image_extensions, pdf_extensions)
        else:
            logger.error(f"Invalid input path: {input_path}")
            return []

        logger.info(f"Processing complete! Generated {len(outputs)} output files.")
        logger.info("=== Document Rotation Process Finished ===")

        return outputs

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return []

# Example usage
if __name__ == "__main__":
    # Example configuration dictionary
    config = {
        'input_path': r'C:\Users\syed.hasnain\Downloads\nputFolder',  # Required
        'output_dir': r'C:\Users\syed.hasnain\Downloads\newImah',  # Optional
        # 'image_extensions': {'.jpg', '.png', '.pdf'},  # Optional - custom extensions
        # 'pdf_extensions': {'.pdf'}  # Optional - custom PDF extensions
    }

    # Process documents
    output_files = rotate_documents(config)
