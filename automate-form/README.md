# PDF Filler for Automation Anywhere 360

This project provides a single Python script, `pdf_filler_final_aa360.py`, for use with Automation Anywhere 360.

It supports two steps:

1. Map a PDF template using a desktop GUI
2. Fill PDFs from Excel using saved JSON mappings

The script is designed for AA360's `Python script: Execute function` action, where only one argument can be passed to the function. Because of that, both functions accept a single list input called `args`.

## Main File

- `pdf_filler_final_aa360.py`

## Functions

### 1. `map_pdf_template(args)`

Use this function to create a mapping JSON for a PDF template.

Expected input list:

```python
[pdf_path]
```

Optional extended input:

```python
[pdf_path, template_id, mappings_dir]
```

Meaning:

- `args[0]` = PDF file path
- `args[1]` = optional template ID
- `args[2]` = optional mappings folder path

Return value:

- Returns the `template_id`
- Example: if the mapping file is `bank.json`, the function returns `bank`

What this function does:

- opens the PDF mapping GUI
- lets you draw field boxes on the PDF
- saves the mapping as a JSON file inside the mappings folder

Important:

- This function opens a GUI, so it must be run on a desktop machine
- It should be used once per PDF template for setup

### 2. `fill_pdfs_from_excel(args)`

Use this function to fill PDFs using Excel data and a folder of mapping JSON files.

Expected input list:

```python
[mappings_dir, excel_path, output_dir]
```

Optional extended input:

```python
[mappings_dir, excel_path, output_dir, row_index]
```

Meaning:

- `args[0]` = folder containing mapping JSON files
- `args[1]` = Excel file path
- `args[2]` = output folder path
- `args[3]` = optional row index (0-based)

Return value:

- Returns the generated PDF file paths as one string joined by `|`

What this function does:

- reads the Excel file
- checks the `template_id` column in each row
- finds the matching JSON file in the mappings folder
- opens the original PDF path stored in that JSON
- fills the fields
- saves the output PDF

## Required Excel Format

Your Excel file must contain a column named:

```text
template_id
```

Example:

| template_id | Full Name   | Account No |
|-------------|-------------|------------|
| axis        | John Doe    | 123456     |
| hdfc        | Mary Thomas | 987654     |

How it works:

- row with `template_id = axis` uses `axis.json`
- row with `template_id = hdfc` uses `hdfc.json`

If the `template_id` cell is blank, that row is skipped.

## Example Folder Structure

```text
pdf_filler/
|-- pdf_filler_final_aa360.py
|-- mappings/
|   |-- axis.json
|   |-- hdfc.json
|-- output/
|-- templates/
|   |-- axis.pdf
|   |-- hdfc.pdf
|-- data_customers_FIXED.xlsx
```

## Automation Anywhere 360 Usage

### Step 1: Open the Python file

Use AA360 `Python script: Open` and select:

```text
pdf_filler_final_aa360.py
```

### Step 2: Run mapping function

Function name:

```text
map_pdf_template
```

Argument passed from AA360:

```python
[pdf_path]
```

Example:

```python
["C:\\Users\\DevaKumar\\OneDrive - Automation Anywhere\\Desktop\\New folder\\pdf_filler\\templates\\axis.pdf"]
```

Returned result:

```text
axis
```

### Step 3: Run fill function

Function name:

```text
fill_pdfs_from_excel
```

Argument passed from AA360:

```python
[mappings_dir, excel_path, output_dir]
```

## Python Dependencies

Install these packages in the same Python environment used by AA360:

```bash
pip install openpyxl pymupdf pillow
```

Modules used:

- `openpyxl`
- `fitz` from `pymupdf`
- `PIL` from `pillow`
- `tkinter` (usually included with standard Python on Windows)

## Important Notes

- `map_pdf_template()` uses a GUI and should not be run on a headless bot runner
- `fill_pdfs_from_excel()` works in batch mode
- the JSON mapping stores the original PDF path
- field names in the mapping should match Excel column names exactly
- output PDFs are saved to the folder you pass in `args[2]`

## Troubleshooting

### Syntax error in AA360

If AA360 shows a bot error, check the bot agent logs such as:

```text
bot_launcher.log
```

### Module not found

If you see errors like:

```text
ModuleNotFoundError: No module named 'openpyxl'
```

or

```text
ModuleNotFoundError: No module named 'fitz'
```

install the missing package in the same Python version used by Automation Anywhere.

### Mapping not found

If a row contains:

```text
template_id = axis
```

then the script expects this file inside the mappings folder:

```text
axis.json
```

### PDF GUI does not open

Run the mapping function on a desktop runner with access to the screen.

## Summary

Use this order in AA360:

1. Run `map_pdf_template([pdf_path])`
2. This creates `template_id.json` and returns `template_id`
3. Run `fill_pdfs_from_excel([mappings_dir, excel_path, output_dir])`
4. The script reads `template_id` from Excel and automatically picks the correct JSON from the folder
