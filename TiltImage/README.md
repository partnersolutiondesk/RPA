# Document Rotation and OCR Processing

This script detects and corrects the orientation of scanned documents (images and PDFs) using **Tesseract OCR**.  
It can process single files or entire directories and supports multiple image formats and PDFs.

## Features
- Automatic orientation detection using Tesseract OCR
- Fallback orientation detection by testing **0°**, **90°**, **180°**, and **270°** rotations
- Image enhancement (contrast and sharpness) for better OCR accuracy
- Supports **JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP**, and **PDF** files
- Batch processing for directories
- Logging of all operations in `processing.log`
- Converts PDF pages to high-resolution images for better OCR

## Requirements
- **Python 3.7+**
- Install dependencies:
  ```bash
  pip install pillow pdf2image pytesseract opencv-python numpy
  ```
- Install Tesseract OCR:
  - **Windows:** [Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage
1. Import and call the `rotate_documents` function with a configuration dictionary:
   ```python
   from document_rotator import rotate_documents

   rotate_documents({
       'input_path': 'path/to/file/or/directory',
       'output_dir': 'rotated_documents'
   })
   ```

2. **Supported formats**:
   - **Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
   - **PDF:** `.pdf`

## Output
- Corrected images are saved with `_corrected` in the filename.
- PDFs are split into separate page images (e.g., `file_page_001_corrected.png`).
- Logs are saved to `output_dir/processing.log`.

## Notes
- **Tesseract must be installed** and available in the system PATH.
- Use high-resolution scans (≥300 DPI) for best results.
- PDF pages are saved as separate images.
