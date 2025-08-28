import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
import logging

try:
    from PIL import Image, ImageEnhance, ImageOps
    import pdf2image
    import pytesseract
    import cv2
    import numpy as np
except ImportError as e:
    raise ImportError(f"Missing required library: {e}. Install with: pip install pillow pdf2image pytesseract opencv-python numpy")

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file in output directory"""
    log_file = output_dir / "processing.log"
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger('document_rotator')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)

    return logger

def preprocess_for_tesseract(image: Image.Image) -> Image.Image:
    """
    Aggressive preprocessing to make text more detectable
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Resize image if it's too small (Tesseract works better with larger images)
    height, width = gray.shape
    if min(height, width) < 300:
        scale_factor = 300 / min(height, width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)

    # Convert back to PIL Image
    processed = Image.fromarray(enhanced, mode='L')

    return processed

def detect_orientation_direct_ocr(image: Image.Image, logger: logging.Logger) -> Tuple[int, float]:
    """
    Skip Tesseract OSD and directly test OCR quality at different orientations
    """
    logger.info("Using direct OCR orientation detection (bypassing Tesseract OSD)")

    best_angle = 0
    best_score = 0

    # Test each rotation
    for angle in [0, 90, 180, 270]:
        try:
            # Rotate image
            if angle == 0:
                test_image = image
            else:

                test_image = image.rotate(-angle, expand=True, fillcolor='white')


            # Preprocess for better OCR
            processed = preprocess_for_tesseract(test_image)

            # Try multiple OCR approaches and combine results
            scores = []

            # Method 1: Basic OCR with word counting
            try:
                text = pytesseract.image_to_string(
                    processed,
                    config='--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '
                ).strip()

                # Count meaningful words (length >= 2, mostly letters)
                words = [w for w in text.split() if len(w) >= 2 and sum(c.isalpha() for c in w) >= len(w)*0.7]
                word_score = len(words) * 2
                scores.append(word_score)

            except:
                scores.append(0)

            # Method 2: OCR with confidence data
            try:
                data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 30]  # Filter low confidence
                conf_score = len(confidences) * 0.5
                scores.append(conf_score)

            except:
                scores.append(0)

            # Method 3: Line detection for text blocks
            try:
                img_array = np.array(processed)
                edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)

                if lines is not None:
                    # Count horizontal-ish lines (likely text lines)
                    horizontal_lines = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle_deg = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                        if angle_deg < 15 or angle_deg > 165:  # Nearly horizontal
                            horizontal_lines += 1

                    line_score = horizontal_lines * 0.3
                    scores.append(line_score)
                else:
                    scores.append(0)

            except:
                scores.append(0)

            # Combine scores
            total_score = sum(scores)

            logger.info(f"Angle {angle}°: word_score={scores[0] if len(scores)>0 else 0:.1f}, "
                        f"conf_score={scores[1] if len(scores)>1 else 0:.1f}, "
                        f"line_score={scores[2] if len(scores)>2 else 0:.1f}, "
                        f"total={total_score:.1f}")

            if total_score > best_score:
                best_score = total_score
                best_angle = angle

        except Exception as e:
            logger.warning(f"Failed to test angle {angle}°: {e}")
            continue

    # Convert score to confidence (0-100)
    confidence = min(best_score * 2, 100) if best_score > 0 else 0

    logger.info(f"Direct OCR result: {best_angle}° rotation needed (confidence: {confidence:.1f})")
    return best_angle, confidence

def detect_orientation_simple_fallback(image: Image.Image, logger: logging.Logger) -> Tuple[int, float]:
    """
    Ultra-simple fallback: just try to read text at each orientation
    """
    logger.info("Using simple text-based orientation detection")

    best_angle = 0
    best_char_count = 0

    for angle in [0, 90, 180, 270]:
        try:
            # Rotate image
            if angle == 0:
                test_image = image
            else:
                test_image = image.rotate(-angle, expand=True, fillcolor='white')

            # Simple enhancement
            if test_image.mode != 'L':
                test_image = test_image.convert('L')
            test_image = ImageOps.autocontrast(test_image)

            # Try to extract text with very permissive settings
            try:
                text = pytesseract.image_to_string(
                    test_image,
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? '
                )

                # Count alphabetic characters
                char_count = sum(1 for c in text if c.isalpha())

                logger.info(f"Angle {angle}°: {char_count} alphabetic characters")

                if char_count > best_char_count:
                    best_char_count = char_count
                    best_angle = angle

            except Exception as ocr_error:
                logger.info(f"OCR failed for angle {angle}°")

        except Exception as e:
            logger.warning(f"Failed to test angle {angle}°: {e}")
            continue

    confidence = min(best_char_count / 10, 50) if best_char_count > 0 else 0
    logger.info(f"Simple fallback result: {best_angle}° rotation (chars: {best_char_count})")

    return best_angle, confidence

def detect_orientation_robust(image: Image.Image, logger: logging.Logger) -> Tuple[int, float]:
    """
    Robust orientation detection that avoids Tesseract OSD issues
    """
    # Skip Tesseract OSD entirely and go straight to direct methods
    angle, confidence = detect_orientation_direct_ocr(image, logger)

    # If confidence is too low, try the simple fallback
    if confidence < 5:
        logger.info("Low confidence, trying simple fallback")
        angle, confidence = detect_orientation_simple_fallback(image, logger)

    return angle, confidence

def rotate_image(image: Image.Image, angle: int, logger: logging.Logger) -> Image.Image:
    """
    Rotate image by specified angle
    """
    if angle == 0:
        logger.info("No rotation needed")
        return image

    rotated = image.rotate(-angle, expand=True, fillcolor='white')
    logger.info(f"Rotated image by {angle}°")
    return rotated

def process_pdf(pdf_path: Path, output_dir: Path, logger: logging.Logger) -> List[Path]:
    """
    Process PDF file: convert to images, detect orientation, and correct
    """
    logger.info(f"Processing PDF: {pdf_path}")

    try:
        # Convert PDF to images with moderate DPI
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=200,
            fmt='PNG'
        )

        output_paths = []

        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")

            # Detect orientation using robust method
            rotation_angle, confidence = detect_orientation_robust(image, logger)

            # Apply rotation if we have any confidence
            if confidence > 1 or rotation_angle != 0:
                corrected_image = rotate_image(image, rotation_angle, logger)
            else:
                logger.info("Very low confidence, keeping original orientation")
                corrected_image = image

            # Save corrected image
            output_filename = f"{pdf_path.stem}_page_{i+1:03d}_corrected.png"
            output_path = output_dir / output_filename
            corrected_image.save(output_path, 'PNG', dpi=(200, 200))
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

        # Handle different image modes
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')

        # Detect orientation using robust method
        rotation_angle, confidence = detect_orientation_robust(image, logger)

        # Apply rotation if we have any confidence
        if confidence > 1 or rotation_angle != 0:
            corrected_image = rotate_image(image, rotation_angle, logger)
        else:
            logger.info("Very low confidence, keeping original orientation")
            corrected_image = image

        # Save corrected image
        output_filename = f"{image_path.stem}_corrected{image_path.suffix}"
        output_path = output_dir / output_filename
        corrected_image.save(output_path, dpi=(200, 200))

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

    supported_files = [f for f in directory_path.iterdir()
                       if f.is_file() and f.suffix.lower() in supported_extensions]

    logger.info(f"Found {len(supported_files)} supported files")

    for file_path in supported_files:
        outputs = process_file(file_path, output_dir, logger, image_extensions, pdf_extensions)
        all_outputs.extend(outputs)

    return all_outputs

def rotate_documents(config: Dict[str, Any]) -> List[Path]:
    """
    Main function to rotate documents based on configuration dictionary
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
    logger.info("=== Robust Document Rotation Process Started ===")
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
        logger.info("=== Robust Document Rotation Process Finished ===")

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