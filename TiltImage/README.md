Document Rotation and OCR Processing



This script detects and corrects the orientation of scanned documents (images and PDFs) using Tesseract OCR. It can process single files or entire directories and supports multiple image formats and PDFs.



Features:

\- Automatic orientation detection using Tesseract OCR

\- Fallback orientation detection by testing 0°, 90°, 180°, and 270° rotations

\- Image enhancement (contrast and sharpness) for better OCR accuracy

\- Supports JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP, and PDF files

\- Batch processing for directories

\- Logging of all operations in processing.log

\- Converts PDF pages to high-resolution images for better OCR



Requirements:

\- Python 3.7+

\- Install dependencies:

&nbsp;   pip install pillow pdf2image pytesseract opencv-python numpy

\- Install Tesseract OCR:

&nbsp;   Windows: https://github.com/UB-Mannheim/tesseract/wiki

&nbsp;   Linux (Debian/Ubuntu): sudo apt install tesseract-ocr

&nbsp;   macOS: brew install tesseract



Usage:

1\. Import and call the rotate\_documents function with a configuration dictionary:

&nbsp;   from document\_rotator import rotate\_documents



&nbsp;   rotate\_documents({

&nbsp;       'input\_path': 'path/to/file/or/directory',

&nbsp;       'output\_dir': 'rotated\_documents'

&nbsp;   })



2\. Supported formats:

&nbsp;   Images: .jpg .jpeg .png .bmp .tiff .tif .webp

&nbsp;   PDF: .pdf



Output:

\- Corrected images are saved with "\_corrected" in the filename.

\- PDFs are split into separate page images (e.g., file\_page\_001\_corrected.png).

\- Logs are saved to output\_dir/processing.log.



Notes:

\- Tesseract must be installed and available in the system PATH.

\- Use high resolution scans (≥300 DPI) for best results.

\- PDF pages are saved as separate images; combining them back into a PDF is not included.



