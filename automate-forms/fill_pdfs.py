
import argparse
import json
import os
import datetime
import openpyxl
import fitz

# DEFAULT_FILL_COLOR = (1, 1, 1)  # Removed white background to look natural
DEFAULT_FONT = "helv"  # PyMuPDF default font (helv = Helvetica)

def main():
    parser = argparse.ArgumentParser(description="Fill PDFs using Excel or JSON data and coordinate mappings.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--excel", help="Path to the Excel data file (.xlsx)")
    input_group.add_argument("--json", help="Path to the JSON data file")
    parser.add_argument("--row", type=int, help="Fill only the specified row (0-indexed)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    # Load data (either Excel or JSON)
    headers = []
    data_rows = []
    if args.excel:
        if not os.path.exists(args.excel):
            print(f"Error: Excel file not found: {args.excel}")
            return
        # Load Excel
        wb = openpyxl.load_workbook(args.excel, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            print("Error: Excel file has no data rows (needs header + at least one data row)")
            return
        headers = [str(h).strip() if h is not None else "" for h in rows[0]]
        data_rows = rows[1:]
    elif args.json:
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found: {args.json}")
            return
        # Load JSON (expects array of objects)
        with open(args.json, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Error: JSON file must contain an array of objects")
            return
        if len(data) < 1:
            print("Error: JSON file has no data rows")
            return
        # Get headers from first object
        headers = list(data[0].keys())
        # Convert objects to list of values in header order
        data_rows = []
        for obj in data:
            row_values = []
            for header in headers:
                val = obj.get(header, "")
                row_values.append(str(val) if val is not None else "")
            data_rows.append(row_values)

    if "template_id" not in headers:
        print("Error: Data file must have a 'template_id' field/column")
        return
    template_id_idx = headers.index("template_id")

    if args.row is not None:
        if 0 <= args.row < len(data_rows):
            data_rows = [data_rows[args.row]]
        else:
            print(f"Error: Row index {args.row} is out of range (0 to {len(data_rows)-1})")
            return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mappings_dir = os.path.join(script_dir, "mappings")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for row_idx, row_data in enumerate(data_rows):
        template_id = row_data[template_id_idx]
        if not template_id:
            print(f"Skipping row {row_idx}: No template_id")
            continue

        mapping_file = os.path.join(mappings_dir, f"{template_id}.json")
        if not os.path.exists(mapping_file):
            print(f"Skipping row {row_idx}: Mapping file not found: {mapping_file}")
            continue

        # Load mapping
        with open(mapping_file, "r") as f:
            mapping = json.load(f)

        # Load PDF
        pdf_path = mapping["pdf_path"]
        if not os.path.exists(pdf_path):
            print(f"Skipping row {row_idx}: PDF template not found: {pdf_path}")
            continue

        doc = fitz.open(pdf_path)
        row_dict = dict(zip(headers, [str(cell) if cell is not None else "" for cell in row_data]))

        for field in mapping["fields"]:
            field_name = field["field_name"]
            value = row_dict.get(field_name, "").strip()
            page_num = field["page"]
            if page_num < 0 or page_num >= len(doc):
                continue
            page = doc[page_num]
            font_family = field.get("font_family", DEFAULT_FONT)
            font_size = field["font_size"]
            field_mode = field.get("mode", "normal")

            if field_mode == "option":
                # Option/Checkbox mode: find the matching option and draw a tick mark
                if not value:
                    continue
                for opt in field["options"]:
                    if opt["option"] == value:
                        # Draw a tick mark in this box
                        x1 = opt["x1"]
                        y1 = opt["y1"]
                        x2 = opt["x2"]
                        y2 = opt["y2"]
                        # Calculate center of the box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        # Calculate box size
                        box_width = x2 - x1
                        box_height = y2 - y1
                        # Make tick mark proportional to box size
                        tick_size = min(box_width, box_height) * 0.6
                        # Draw tick mark (two lines: down-right then up-right)
                        # First line: from top-left of tick area to bottom-middle
                        start1 = (center_x - tick_size * 0.4, center_y - tick_size * 0.2)
                        mid = (center_x, center_y + tick_size * 0.3)
                        # Second line: from mid to top-right of tick area
                        end2 = (center_x + tick_size * 0.6, center_y - tick_size * 0.4)
                        page.draw_line(start1, mid, color=(0, 0, 0), width=2)
                        page.draw_line(mid, end2, color=(0, 0, 0), width=2)
                        break
            elif "x1" in field and "x2" in field and "y1" in field and "y2" in field:
                # Normal or charwise mode
                x1 = field["x1"]
                y1 = field["y1"]
                x2 = field["x2"]
                y2 = field["y2"]
                rect_width = x2 - x1
                rect_height = y2 - y1

                if field_mode == "charwise":
                    # Character-wise mode: stamp each character individually
                    if not value:
                        continue
                    num_boxes = field.get("num_boxes", len(value))
                    num_chars = len(value)
                    if num_boxes == 0:
                        continue
                    # Calculate spacing between character box centers
                    char_spacing = rect_width / num_boxes
                    # Calculate starting x (center of first box)
                    start_x = x1 + char_spacing / 2
                    # Y position: center vertically
                    y = y1 + rect_height / 2 + font_size / 3

                    for i in range(min(num_chars, num_boxes)):
                        char = value[i]
                        char_x = start_x + (i * char_spacing)
                        # Center this character
                        char_width = fitz.get_text_length(char, fontname=font_family, fontsize=font_size)
                        centered_char_x = char_x - (char_width / 2)
                        page.insert_text((centered_char_x, y), char, fontname=font_family, fontsize=font_size, color=(0, 0, 0))
                else:
                    # Normal single-text mode
                    if not value:
                        continue
                    text_width = fitz.get_text_length(value, fontname=font_family, fontsize=font_size)
                    x = x1 + (rect_width - text_width) / 2
                    y = y1 + rect_height / 2 + font_size / 3
                    page.insert_text((x, y), value, fontname=font_family, fontsize=font_size, color=(0, 0, 0))
            else:
                # Old format: use existing x,y (only single-text mode)
                if not value:
                    continue
                x = field["x"]
                y = field["y"]
                page.insert_text((x, y), value, fontname=font_family, fontsize=font_size, color=(0, 0, 0))

        if args.dry_run:
            print(f"Dry run: Would save filled PDF for template '{template_id}', row {row_idx}")
            doc.close()
            continue

        # Save output
        hint = row_dict.get("Full Name", row_dict.get("customer_name", f"row{row_idx}")).replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{template_id}_row{row_idx}_{hint}_{timestamp}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        doc.save(output_path)
        print(f"Saved: {output_path}")
        doc.close()

    if args.excel:
        wb.close()

if __name__ == "__main__":
    main()
