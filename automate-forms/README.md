# Automate Form PDF Filler

This repository contains the standard Python version of the PDF filling workflow.

This README explains the code flow using:

- `tools/map_template.py`
- `tools/batch_map_all_templates.py`
- `fill_pdfs.py`

This README does not use or explain `pdf_filler_final_aa360.py`.

## What the project does

The project fills PDF templates using coordinate mappings saved in JSON files.

The workflow is:

1. Put blank PDF templates in the `templates/` folder
2. Create mapping JSON files for those templates
3. Prepare Excel or JSON data with a `template_id`
4. Run `fill_pdfs.py`
5. The script matches each data row to the correct mapping and generates filled PDFs

## Main files

### `tools/map_template.py`

This is the manual PDF mapping tool.

What it does:

- opens a PDF in a desktop GUI
- lets you draw boxes where text should be filled
- supports:
  - normal text fields
  - character-wise fields
  - option or checkbox fields
- saves the result as a JSON mapping file

Example command:

```bash
python tools/map_template.py --pdf "templates/axis.pdf" --id axis
```

What gets created:

```text
mappings/axis.json
```

### `tools/batch_map_all_templates.py`

This script helps you map many PDFs one after another.

What it does:

- looks inside the `templates/` folder
- finds all `.pdf` files
- generates a template ID from the file name
- opens `tools/map_template.py` for each PDF
- waits for you to finish one mapping before moving to the next

Example command:

```bash
python tools/batch_map_all_templates.py
```

If `templates/` contains:

```text
axis.pdf
hdfc.pdf
icici.pdf
```

then the script will try to create:

```text
mappings/axis.json
mappings/hdfc.json
mappings/icici.json
```

### `fill_pdfs.py`

This is the main filling script.

What it does:

- accepts Excel or JSON data
- requires a `template_id` field in the data
- loads the correct mapping JSON from the `mappings/` folder
- opens the original PDF template path stored in that mapping
- fills the PDF using the row data
- saves the result into the `output/` folder

It supports:

- normal text
- character-wise text
- option or checkbox ticking

## Folder structure

Expected structure:

```text
automate-form/
|-- fill_pdfs.py
|-- templates/
|   |-- axis.pdf
|   |-- hdfc.pdf
|-- mappings/
|   |-- axis.json
|   |-- hdfc.json
|-- output/
|-- tools/
|   |-- map_template.py
|   |-- batch_map_all_templates.py
```

## How the code works

### Step 1: Mapping

You first create a mapping for each blank PDF template.

There are two ways:

#### Option A: Map one PDF manually

```bash
python tools/map_template.py --pdf "templates/axis.pdf" --id axis
```

#### Option B: Map all PDFs one by one

```bash
python tools/batch_map_all_templates.py
```

During mapping:

- you type the field name
- the field name should match the Excel or JSON key exactly
- you drag a box on the PDF
- the script stores the coordinates, font, size, mode, and page number

That data is saved in a JSON file inside `mappings/`.

### Step 2: Prepare data

Your data can be:

- Excel `.xlsx`
- JSON file containing an array of objects

The data must include:

```text
template_id
```

Example Excel columns:

| template_id | Full Name | Account No | Gender |
| ----------- | --------- | ---------- | ------ |
| axis        | John Doe  | 12345      | Male   |
| hdfc        | Mary      | 67890      | Female |

Example JSON:

```json
[
  {
    "template_id": "axis",
    "Full Name": "John Doe",
    "Account No": "12345",
    "Gender": "Male"
  },
  {
    "template_id": "hdfc",
    "Full Name": "Mary",
    "Account No": "67890",
    "Gender": "Female"
  }
]
```

### Step 3: Fill PDFs

Run with Excel:

```bash
python fill_pdfs.py --excel "data.xlsx"
```

Run with JSON:

```bash
python fill_pdfs.py --json "data.json"
```

Run only one row:

```bash
python fill_pdfs.py --excel "data.xlsx" --row 0
```

Preview only without saving:

```bash
python fill_pdfs.py --excel "data.xlsx" --dry-run
```

## How `fill_pdfs.py` chooses the correct PDF

For each row:

1. read `template_id`
2. look for `mappings/<template_id>.json`
3. load the mapping file
4. read `pdf_path` stored inside that mapping
5. open the PDF
6. fill each mapped field using the row data
7. save the final PDF in `output/`

Example:

- row has `template_id = axis`
- script loads `mappings/axis.json`
- mapping contains the original `pdf_path`
- output PDF gets created for that row

## Field modes

### Normal mode

Used for normal text fields.

The script:

- reads the mapped rectangle
- calculates text width
- centers the value inside the box

### Character-wise mode

Used when each character should go into a separate box.

The script:

- divides the mapped area into equal sections
- places one character in each section

### Option or checkbox mode

Used for checkboxes or radio-style options.

The script:

- checks the row value
- finds the matching option in the mapping
- draws a tick mark in that option box

## Output file naming

Generated files are saved in `output/`.

The output name is based on:

- `template_id`
- row number
- `Full Name` or `customer_name` if present
- timestamp

Example:

```text
axis_row0_John_Doe_20260715_101530.pdf
```

## Python requirements

Install these packages:

```bash
pip install openpyxl pymupdf pillow
```

Used modules:

- `openpyxl`
- `fitz` from `pymupdf`
- `PIL` from `pillow`
- `tkinter`

Note:

- `tkinter` is usually already included with Windows Python
- mapping requires a desktop screen because it opens a GUI

## Important rules

- PDF file names and mapping names should stay consistent
- `template_id` in data must match the JSON file name
- mapped field names must match Excel column names or JSON keys exactly
- if a mapping file is missing, that row is skipped
- if the original PDF path inside the mapping is missing, that row is skipped

## Troubleshooting

### No output generated

Check:

- the data file contains `template_id`
- matching JSON files exist in `mappings/`
- the PDF file paths stored in those JSON files are valid

### Mapping file not found

If a row has:

```text
template_id = axis
```

then this file must exist:

```text
mappings/axis.json
```

### GUI does not open

`tools/map_template.py` and `tools/batch_map_all_templates.py` need a desktop environment.

### Wrong field filled

Check:

- field name in mapping
- Excel column name or JSON key
- correct page selection during mapping
- field box position

## Quick start

### Map one template

```bash
python tools/map_template.py --pdf "templates/axis.pdf" --id axis
```

### Fill from Excel

```bash
python fill_pdfs.py --excel "data.xlsx"
```

### Fill from JSON

```bash
python fill_pdfs.py --json "data.json"
```

## Summary

Use:

- `tools/map_template.py` to create one mapping
- `tools/batch_map_all_templates.py` to create many mappings
- `fill_pdfs.py` to generate filled PDFs from Excel or JSON data

The main rule is simple:

- `template_id` in your data must match the JSON file name in `mappings/`
