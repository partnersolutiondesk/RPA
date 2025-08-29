# Document Rotation and OCR Correction

This tool automatically detects and corrects the orientation of scanned documents (images and PDFs).  
It bypasses Tesseract’s built-in **OSD (Orientation & Script Detection)** and instead uses **OCR-based heuristics** to determine the correct rotation.

---

## ✨ Features
- ✅ Works on **images** (`.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp`) and **PDFs**.  
- ✅ **Bypasses Tesseract OSD** — uses custom orientation detection with OCR + image analysis.  
- ✅ Multiple rotation checks (0°, 90°, 180°, 270°) with scoring:
  - OCR word detection  
  - OCR confidence values  
  - Text line structure analysis  
- ✅ Fallback character-counting method for very poor scans.  
- ✅ Image preprocessing:
  - Grayscale conversion  
  - Noise reduction (bilateral filter)  
  - Contrast enhancement (CLAHE)  
  - Resizing small images for better OCR results  
- ✅ Detailed **logging** (`processing.log`) in the output directory.  
- ✅ Batch processing of files and directories.  

---

## 📦 Requirements
Install dependencies with:

```bash
pip install pillow pdf2image pytesseract opencv-python numpy
```

You also need:
- **Tesseract OCR** installed on your system.  
  - [Installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)  
  - Make sure it’s in your PATH (`tesseract --version` should work).  
- **Poppler** (for PDF to image conversion via `pdf2image`).  

---

## 🚀 Usage

### Example: Run on a single file
```python
from rotate_documents import rotate_documents

config = {
    'input_path': 'path/to/document.pdf',
    'output_dir': 'corrected_documents'
}

results = rotate_documents(config)
print("Corrected files:", results)
```

### Example: Run on a directory
```python
config = {
    'input_path': 'scanned_docs/',   # directory of images/PDFs
    'output_dir': 'corrected_docs'
}

results = rotate_documents(config)
```

## 📂 Output
- Corrected images are saved in the configured output directory.  
- PDFs are split into per-page **PNG images**.  
- A `processing.log` file is generated with detailed rotation decisions.  

---

## ⚙️ Configuration Options
You can pass a `config` dictionary with:

| Key               | Default Value | Description |
|-------------------|---------------|-------------|
| `input_path`      | (required)    | File or directory to process |
| `output_dir`      | `rotated_documents` | Folder for corrected files |
| `image_extensions`| `{.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp}` | Supported image types |
| `pdf_extensions`  | `{.pdf}`      | Supported PDF types |

---

## 🔍 How It Works
1. Load input (image/PDF).  
2. Convert to image (if PDF).  
3. Preprocess (grayscale, resize, denoise, enhance).  
4. Test **0°, 90°, 180°, 270°**:
   - Extract text via OCR.  
   - Score based on word count, OCR confidence, line detection.  
5. Pick best orientation → Rotate.  
6. Save corrected result.  

Unlike Tesseract’s OSD, this approach works even when `osd.traineddata` is missing.

---


```

---

## 📜 License
Global Solution Desk – feel free to use and modify.  
Author: Syed
