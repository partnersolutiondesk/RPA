
import os
import sys
import json
import datetime
import openpyxl
import fitz
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ----------------------
# Configuration (Internal)
# ----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAPPINGS_DIR = os.path.join(PROJECT_ROOT, "mappings")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_FONT = "helv"

os.makedirs(MAPPINGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------
# Helper: Normalize AA360 single-input argument list
# ----------------------
def _normalize_args(args):
    """Accept AA360 list input and keep basic backward compatibility."""
    if isinstance(args, (list, tuple)):
        return list(args)
    if args is None:
        return []
    return [args]


# ----------------------
# Function 1: Map PDF Template (AA360 entry point)
# ----------------------
def map_pdf_template(args):
    """
    AA360 Function 1: Map a PDF template.

    Expected AA360 input list:
        args[0] = pdf_path
        args[1] = template_id (optional)
        args[2] = mappings_dir (optional)

    Returns:
        template_id (string, for example "axis")
    """
    args = _normalize_args(args)
    if len(args) < 1 or not str(args[0]).strip():
        raise ValueError("map_pdf_template expects args[0] = pdf_path")

    pdf_path = str(args[0]).strip()
    template_id = str(args[1]).strip() if len(args) > 1 and args[1] is not None and str(args[1]).strip() else None
    mappings_dir = str(args[2]).strip() if len(args) > 2 and args[2] is not None and str(args[2]).strip() else None

    # Use default mappings dir if not provided
    if not mappings_dir:
        mappings_dir = MAPPINGS_DIR
    os.makedirs(mappings_dir, exist_ok=True)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Auto-generate template ID if not provided
    if not template_id:
        pdf_filename = os.path.basename(pdf_path)
        template_id = os.path.splitext(pdf_filename)[0].lower().replace(" ", "_")

    # Load PDF
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise ValueError("PDF has no pages")

    # Get page size from first page
    first_page = doc[0]
    page_size_pts = [first_page.rect.width, first_page.rect.height]
    fields = []
    undo_stack = []
    redo_stack = []
    # ----------------------
    # FULL GUI SETUP (ORIGINAL)
    # ----------------------
    root = tk.Tk()
    root.title(f"Map Template: {template_id}")
    root.geometry("1200x850")

    current_page = 0
    zoom = 1.0
    photo = None
    img_width = 0
    img_height = 0
    drag_start = None
    current_drag_rect = None
    current_small_boxes = []
    current_option_boxes = []
    current_field_name = None

    # Instructions Frame
    instructions_frame = ttk.LabelFrame(root, text="📋 Step-by-Step Instructions", padding="10")
    instructions_frame.pack(fill=tk.X, padx=10, pady=5)
    instructions_text = (
        "1. Enter the Full Screen Mode\n"
        "2. Type a Field Name (must match Excel column header exactly! e.g., 'Full Name')\n"
        "3. Adjust Font (Default: 'helv' for printed, 'tiro-italic' for handwritten look)\n"
        "4. Click & DRAG to draw a box around the field area (instead of a dot)\n"
        "5. Use Ctrl+Z or 'Undo Last' to undo\n"
        "6. Use Previous/Next to switch pages\n"
        "7. When done, click Save & Exit"
    )
    ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT).pack(anchor=tk.W)

    # PDF Canvas with scrollbars
    pdf_frame = ttk.Frame(root, padding="10")
    pdf_frame.pack(fill=tk.BOTH, expand=True)
    v_scrollbar = ttk.Scrollbar(pdf_frame, orient=tk.VERTICAL)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar = ttk.Scrollbar(pdf_frame, orient=tk.HORIZONTAL)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas = tk.Canvas(pdf_frame, bg="white", xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
    canvas.pack(fill=tk.BOTH, expand=True)
    v_scrollbar.config(command=canvas.yview)
    h_scrollbar.config(command=canvas.xview)

    # Control Frame
    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill=tk.X)

    # Row 0: Page nav, zoom, save/exit
    ttk.Label(control_frame, text="Page:").grid(row=0, column=0, padx=5, sticky=tk.W)
    page_var = tk.StringVar(value=f"{current_page + 1}/{page_count}")
    ttk.Label(control_frame, textvariable=page_var).grid(row=0, column=1, padx=5, sticky=tk.W)

    def change_page(delta):
        nonlocal current_page
        new_page = current_page + delta
        if 0 <= new_page < page_count:
            current_page = new_page
            render_page()

    ttk.Button(control_frame, text="Previous", command=lambda: change_page(-1)).grid(row=0, column=2, padx=5, sticky=tk.W)
    ttk.Button(control_frame, text="Next", command=lambda: change_page(1)).grid(row=0, column=3, padx=5, sticky=tk.W)

    ttk.Label(control_frame, text="Zoom:").grid(row=0, column=4, padx=5, sticky=tk.W)

    def zoom_in():
        nonlocal zoom
        zoom = min(zoom + 0.25, 3.0)
        render_page()

    def zoom_out():
        nonlocal zoom
        zoom = max(zoom - 0.25, 0.5)
        render_page()

    ttk.Button(control_frame, text="-", width=3, command=zoom_out).grid(row=0, column=5, padx=2, sticky=tk.W)
    ttk.Button(control_frame, text="+", width=3, command=zoom_in).grid(row=0, column=6, padx=2, sticky=tk.W)

    def save_and_exit():
        if len(fields) == 0:
            confirm = messagebox.askyesno("⚠️ No Fields Mapped", "You haven't mapped any fields. Save anyway?")
            if not confirm:
                return
        mapping = {
            "template_id": template_id,
            "pdf_path": pdf_path,
            "page_count": page_count,
            "page_size_pts": page_size_pts,
            "fields": fields
        }
        mapping_file = os.path.join(mappings_dir, f"{template_id}.json")
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)
        messagebox.showinfo("✅ Success!", f"Mapping saved to {mapping_file}")
        doc.close()
        root.destroy()

    save_btn = ttk.Button(control_frame, text="✅ Save & Exit", command=save_and_exit)
    save_btn.grid(row=0, column=10, pady=5, padx=(20,5), sticky=tk.E)
    control_frame.columnconfigure(10, weight=1)

    # Row 1: Field name, font, size, undo/redo
    ttk.Label(control_frame, text="Field Name (exact Excel column):").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
    field_name_var = tk.StringVar()
    field_name_entry = ttk.Entry(control_frame, textvariable=field_name_var, width=25)
    field_name_entry.grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W, pady=5)
    field_name_entry.focus()

    ttk.Label(control_frame, text="Font:").grid(row=1, column=3, padx=5, sticky=tk.W, pady=5)
    font_family_var = tk.StringVar(value="helv")
    font_options = ["helv", "helv-oblique", "helv-bold", "cour", "cour-oblique", "cour-bold", "tiro", "tiro-italic", "tiro-bold"]
    font_menu = ttk.Combobox(control_frame, textvariable=font_family_var, values=font_options, state="readonly", width=15)
    font_menu.grid(row=1, column=4, columnspan=2, padx=5, sticky=tk.W, pady=5)

    ttk.Label(control_frame, text="Font Size:").grid(row=1, column=6, padx=5, sticky=tk.W, pady=5)
    font_size_var = tk.IntVar(value=11)
    ttk.Spinbox(control_frame, from_=5, to=20, textvariable=font_size_var, width=5).grid(row=1, column=7, padx=5, sticky=tk.W, pady=5)

    def undo_last():
        if not fields:
            messagebox.showinfo("Info", "Nothing to undo!")
            return
        last_field = fields.pop()
        undo_stack.append(last_field)
        redo_stack.clear()
        update_fields_list()
        render_page()

    ttk.Button(control_frame, text="Undo", command=undo_last).grid(row=1, column=8, padx=5, sticky=tk.W, pady=5)

    def redo_last():
        if not undo_stack:
            messagebox.showinfo("Info", "Nothing to redo!")
            return
        last_undone = undo_stack.pop()
        fields.append(last_undone)
        update_fields_list()
        render_page()

    ttk.Button(control_frame, text="Redo", command=redo_last).grid(row=1, column=9, padx=5, sticky=tk.W, pady=5)

    # Row 2: Mode selection
    mode_var = tk.StringVar(value="normal")
    mode_frame = ttk.Frame(control_frame)
    mode_frame.grid(row=2, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Radiobutton(mode_frame, text="Normal", variable=mode_var, value="normal").pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(mode_frame, text="Character-wise", variable=mode_var, value="charwise").pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(mode_frame, text="Option/Checkbox", variable=mode_var, value="option").pack(side=tk.LEFT, padx=5)

    # Row 3: Mode-specific controls
    num_boxes_frame = ttk.Frame(control_frame)
    num_boxes_frame.grid(row=3, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Label(num_boxes_frame, text="Number of Boxes:").pack(side=tk.LEFT, padx=5)
    num_boxes_var = tk.IntVar(value=10)
    num_boxes_spinbox = ttk.Spinbox(num_boxes_frame, from_=1, to=1000, textvariable=num_boxes_var, width=5, state="normal")
    num_boxes_spinbox.pack(side=tk.LEFT, padx=5)

    option_frame = ttk.Frame(control_frame)
    option_frame.grid(row=3, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Label(option_frame, text="Option Label:").pack(side=tk.LEFT, padx=5)
    option_label_var = tk.StringVar(value="Y")
    option_label_entry = ttk.Entry(option_frame, textvariable=option_label_var, width=10)
    option_label_entry.pack(side=tk.LEFT, padx=5)

    def add_current_as_option():
        nonlocal current_field_name
        if not current_option_boxes:
            messagebox.showwarning("Warning", "First drag a box!")
            return
        last_box = current_option_boxes[-1]
        option_label = option_label_var.get().strip()
        if not option_label:
            messagebox.showwarning("Warning", "Enter an option label!")
            return
        last_box["option"] = option_label
        current_options_label_var.set(", ".join([box["option"] for box in current_option_boxes]))
        nonlocal drag_start, current_drag_rect, current_small_boxes
        drag_start = None
        canvas.delete(current_drag_rect)
        for box in current_small_boxes:
            canvas.delete(box)
        current_drag_rect = None
        current_small_boxes = []
        render_page()
        option_label_var.set("")

    add_option_btn = ttk.Button(option_frame, text="Add Option Box", command=add_current_as_option)
    add_option_btn.pack(side=tk.LEFT, padx=5)

    def finish_option_field():
        nonlocal current_option_boxes, current_field_name
        if not current_option_boxes:
            messagebox.showwarning("Warning", "No options added!")
            return
        field_name = current_field_name if current_field_name else field_name_var.get().strip()
        if not field_name:
            messagebox.showwarning("Warning", "Enter a field name!")
            return
        field_dict = {
            "field_name": field_name,
            "page": current_page,
            "font_family": font_family_var.get(),
            "font_size": font_size_var.get(),
            "mode": "option",
            "options": current_option_boxes
        }
        fields.append(field_dict)
        redo_stack.clear()
        current_option_boxes = []
        current_field_name = None
        current_options_label_var.set("")
        update_fields_list()
        render_page()
        field_name_var.set("")
        field_name_entry.focus()

    finish_options_btn = ttk.Button(option_frame, text="Finish Field", command=finish_option_field)
    finish_options_btn.pack(side=tk.LEFT, padx=5)

    def cancel_option_field():
        nonlocal current_option_boxes, current_field_name, drag_start, current_drag_rect, current_small_boxes
        current_option_boxes = []
        current_field_name = None
        current_options_label_var.set("")
        drag_start = None
        if current_drag_rect:
            canvas.delete(current_drag_rect)
        for box in current_small_boxes:
            canvas.delete(box)
        current_drag_rect = None
        current_small_boxes = []
        render_page()
        field_name_var.set("")
        field_name_entry.focus()

    cancel_options_btn = ttk.Button(option_frame, text="Cancel", command=cancel_option_field)
    cancel_options_btn.pack(side=tk.LEFT, padx=5)
    current_options_label_var = tk.StringVar(value="")
    ttk.Label(option_frame, text="Options:").pack(side=tk.LEFT, padx=5)
    current_options_label = ttk.Label(option_frame, textvariable=current_options_label_var)
    current_options_label.pack(side=tk.LEFT, padx=5)

    def toggle_mode_controls():
        mode = mode_var.get()
        if mode == "charwise":
            num_boxes_frame.grid()
            option_frame.grid_remove()
        elif mode == "option":
            num_boxes_frame.grid_remove()
            option_frame.grid()
        else:
            num_boxes_frame.grid_remove()
            option_frame.grid_remove()

    toggle_mode_controls()
    mode_var.trace_add("write", lambda *args: toggle_mode_controls())

    # Row 4-6: Fields list
    ttk.Label(control_frame, text="Mapped Fields:").grid(row=4, column=0, columnspan=10, sticky=tk.W, pady=(10,5))
    fields_listbox = tk.Listbox(control_frame, height=5)
    fields_listbox.grid(row=5, column=0, columnspan=9, sticky=tk.EW, padx=5, pady=5)
    scrollbar = ttk.Scrollbar(control_frame, orient=tk.VERTICAL, command=fields_listbox.yview)
    scrollbar.grid(row=5, column=9, sticky=tk.NS)
    fields_listbox.config(yscrollcommand=scrollbar.set)
    control_frame.columnconfigure(8, weight=1)

    def delete_selected_field():
        selected = fields_listbox.curselection()
        if selected:
            idx = selected[0]
            deleted_field = fields.pop(idx)
            undo_stack.append(deleted_field)
            redo_stack.clear()
            update_fields_list()
            render_page()

    ttk.Button(control_frame, text="Delete Selected", command=delete_selected_field).grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Render page function
    def render_page():
        nonlocal photo, img_width, img_height
        page = doc[current_page]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_width, img_height = pix.width, pix.height
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Draw existing fields
        for idx, field in enumerate(fields):
            if field["page"] != current_page:
                continue
            field_mode = field.get("mode", "normal")
            if field_mode == "option":
                for opt_idx, opt in enumerate(field["options"]):
                    x1, y1 = opt["x1"]*zoom, opt["y1"]*zoom
                    x2, y2 = opt["x2"]*zoom, opt["y2"]*zoom
                    canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2, tags=f"field_{idx}")
                    canvas.create_text((x1+x2)/2, (y1+y2)/2, text=opt["option"], fill="darkgreen", font=("Arial",10,"bold"), tags=f"field_{idx}")
                first_opt = field["options"][0]
                fx1, fy1 = first_opt["x1"]*zoom, first_opt["y1"]*zoom
                canvas.create_text(fx1, fy1-10, text=field["field_name"], fill="darkblue", font=("Arial",10,"bold"), anchor=tk.W, tags=f"field_{idx}")
            else:
                x1, y1 = field["x1"]*zoom, field["y1"]*zoom
                x2, y2 = field["x2"]*zoom, field["y2"]*zoom
                canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags=f"field_{idx}")
                canvas.create_text((x1+x2)/2, y1-10, text=field["field_name"], fill="darkblue", font=("Arial",10,"bold"), tags=f"field_{idx}")
                if field_mode == "charwise" and "num_boxes" in field:
                    num_boxes = field["num_boxes"]
                    rect_width = x2 - x1
                    box_width = rect_width / num_boxes
                    for i in range(num_boxes):
                        small_x1 = x1 + i * box_width
                        small_x2 = small_x1 + box_width
                        canvas.create_rectangle(small_x1, y1, small_x2, y2, outline="orange", width=1, tags=f"field_{idx}")

        # Draw current option boxes
        for opt_idx, opt in enumerate(current_option_boxes):
            x1, y1 = opt["x1"]*zoom, opt["y1"]*zoom
            x2, y2 = opt["x2"]*zoom, opt["y2"]*zoom
            outline_color = "blue" if not opt["option"] else "green"
            canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2, dash=(5,5) if not opt["option"] else None)
            if opt["option"]:
                canvas.create_text((x1+x2)/2, (y1+y2)/2, text=opt["option"], fill="darkgreen", font=("Arial",10,"bold"))

        page_var.set(f"{current_page + 1}/{page_count}")

    def update_fields_list():
        fields_listbox.delete(0, tk.END)
        for idx, field in enumerate(fields):
            fields_listbox.insert(tk.END, f"{field['field_name']} (Page {field['page'] + 1})")

    # Drag events
    def on_mouse_down(event):
        nonlocal drag_start, current_drag_rect, current_small_boxes
        field_name = field_name_var.get().strip()
        if not field_name:
            messagebox.showwarning("⚠️ Missing Field Name", "First type a Field Name!")
            field_name_entry.focus()
            return
        x_start = canvas.canvasx(event.x)
        y_start = canvas.canvasy(event.y)
        drag_start = (x_start, y_start)
        current_small_boxes = []
        current_drag_rect = canvas.create_rectangle(x_start, y_start, x_start, y_start, outline="blue", width=2, dash=(5,5))

    def on_mouse_drag(event):
        nonlocal current_drag_rect, drag_start, current_small_boxes
        if drag_start is None or current_drag_rect is None:
            return
        x_curr = canvas.canvasx(event.x)
        y_curr = canvas.canvasy(event.y)
        x1 = min(drag_start[0], x_curr)
        y1 = min(drag_start[1], y_curr)
        x2 = max(drag_start[0], x_curr)
        y2 = max(drag_start[1], y_curr)
        canvas.coords(current_drag_rect, drag_start[0], drag_start[1], x_curr, y_curr)
        for box in current_small_boxes:
            canvas.delete(box)
        current_small_boxes = []
        if mode_var.get() == "charwise":
            num_boxes = num_boxes_var.get()
            rect_width = x2 - x1
            if rect_width > 0 and num_boxes >0:
                box_width = rect_width / num_boxes
                for i in range(num_boxes):
                    small_x1 = x1 + i * box_width
                    small_x2 = small_x1 + box_width
                    small_box = canvas.create_rectangle(small_x1, y1, small_x2, y2, outline="orange", width=1, dash=(2,2))
                    current_small_boxes.append(small_box)

    def on_mouse_up(event):
        nonlocal drag_start, current_drag_rect, current_small_boxes, current_option_boxes, current_field_name
        mode = mode_var.get()
        if drag_start is None or current_drag_rect is None:
            return
        x_start, y_start = drag_start
        x_end = canvas.canvasx(event.x)
        y_end = canvas.canvasy(event.y)
        x1 = min(x_start, x_end)
        y1 = min(y_start, y_end)
        x2 = max(x_start, x_end)
        y2 = max(y_start, y_end)
        x1_pts = x1 / zoom
        y1_pts = y1 / zoom
        x2_pts = x2 / zoom
        y2_pts = y2 / zoom

        if mode == "option":
            if not current_option_boxes:
                current_field_name = field_name_var.get().strip()
                if not current_field_name:
                    messagebox.showwarning("⚠️ Missing Field Name", "First type a Field Name!")
                    field_name_entry.focus()
                    drag_start = None
                    canvas.delete(current_drag_rect)
                    for box in current_small_boxes:
                        canvas.delete(box)
                    current_drag_rect = None
                    current_small_boxes = []
                    return
            current_option_boxes.append({
                "x1": x1_pts,
                "y1": y1_pts,
                "x2": x2_pts,
                "y2": y2_pts,
                "option": ""
            })
            drag_start = None
            canvas.delete(current_drag_rect)
            for box in current_small_boxes:
                canvas.delete(box)
            current_drag_rect = None
            current_small_boxes = []
            render_page()
            option_label_entry.focus()
        else:
            field_dict = {
                "field_name": field_name_var.get().strip(),
                "page": current_page,
                "x1": x1_pts,
                "y1": y1_pts,
                "x2": x2_pts,
                "y2": y2_pts,
                "font_family": font_family_var.get(),
                "font_size": font_size_var.get(),
                "mode": "normal"
            }
            if mode_var.get() == "charwise":
                field_dict["mode"] = "charwise"
                field_dict["num_boxes"] = num_boxes_var.get()
            fields.append(field_dict)
            redo_stack.clear()
            drag_start = None
            canvas.delete(current_drag_rect)
            for box in current_small_boxes:
                canvas.delete(box)
            current_drag_rect = None
            current_small_boxes = []
            update_fields_list()
            render_page()
            field_name_var.set("")
            field_name_entry.focus()

    # Bind events
    canvas.bind("<Button-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    def on_mouse_wheel(event):
        nonlocal zoom
        if event.delta:
            zoom_step = 0.25 if event.delta > 0 else -0.25
        else:
            zoom_step = 0.25 if event.num == 4 else -0.25
        zoom = max(0.5, min(zoom + zoom_step, 3.0))
        render_page()

    root.bind("<MouseWheel>", on_mouse_wheel)
    root.bind("<Button-4>", on_mouse_wheel)
    root.bind("<Button-5>", on_mouse_wheel)

    def on_ctrl_z(event):
        undo_last()

    root.bind("<Control-z>", on_ctrl_z)

    def on_ctrl_y(event):
        redo_last()

    root.bind("<Control-y>", on_ctrl_y)

    render_page()
    root.mainloop()
    return template_id


# ----------------------
# Helper Function: Load Single Mapping
# ----------------------
def _load_mapping(mapping_ref, mappings_dir=None):
    """Internal: Load a saved mapping JSON file from path or template_id."""
    mapping_ref = str(mapping_ref).strip()
    if not mappings_dir:
        mappings_dir = MAPPINGS_DIR

    if mapping_ref.lower().endswith(".json") or os.path.exists(mapping_ref):
        mapping_file = mapping_ref
    else:
        mapping_file = os.path.join(mappings_dir, f"{mapping_ref}.json")

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    with open(mapping_file, "r") as f:
        return json.load(f)


def _find_template_id_column(headers):
    """Find the Excel column used to identify which mapping JSON to load."""
    normalized_headers = [str(header).strip().lower() for header in headers]
    for candidate in ["template_id", "templateid", "mapping_name", "json_name"]:
        if candidate in normalized_headers:
            return normalized_headers.index(candidate)
    raise ValueError("Excel file must contain a 'template_id' column")


# ----------------------
# Function 2: Fill PDFs from Excel (AA360 entry point)
# ----------------------
def fill_pdfs_from_excel(args):
    """
    AA360 Function 2: Fill PDFs from mappings folder and Excel file.

    Expected AA360 input list:
        args[0] = mappings_dir
        args[1] = excel_path
        args[2] = output_dir
        args[3] = row_index (optional, 0-based)

    Returns:
        Output PDF paths joined by "|"
    """
    args = _normalize_args(args)
    if len(args) < 2:
        raise ValueError("fill_pdfs_from_excel expects args[0] = mappings_dir, args[1] = excel_path")

    mappings_dir = str(args[0]).strip() if args[0] is not None else ""
    excel_path = str(args[1]).strip()
    output_dir = str(args[2]).strip() if len(args) > 2 and args[2] is not None and str(args[2]).strip() else OUTPUT_DIR
    row_index = None
    if len(args) > 3 and args[3] is not None and str(args[3]).strip() != "":
        row_index = int(args[3])

    if not mappings_dir:
        mappings_dir = MAPPINGS_DIR

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(mappings_dir):
        raise FileNotFoundError(f"Mappings folder not found: {mappings_dir}")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if len(rows) < 2:
        wb.close()
        raise ValueError("Excel file needs at least header + one data row")

    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    template_id_idx = _find_template_id_column(headers)

    data_rows = rows[1:]
    if row_index is not None:
        if not (0 <= row_index < len(data_rows)):
            wb.close()
            raise IndexError(f"Row index {row_index} out of range (0 to {len(data_rows)-1})")
        data_rows = [data_rows[row_index]]

    output_paths = []
    for row_idx, row_data in enumerate(data_rows):
        row_dict = dict(zip(headers, [str(cell) if cell is not None else "" for cell in row_data]))
        template_id = row_dict.get(headers[template_id_idx], "").strip()
        if not template_id:
            print(f"Skipping row {row_idx}: template_id is blank")
            continue

        mapping = _load_mapping(template_id, mappings_dir=mappings_dir)
        pdf_path = mapping.get("pdf_path")
        if not pdf_path:
            wb.close()
            raise ValueError(f"Mapping '{template_id}' does not contain 'pdf_path'")
        doc = fitz.open(pdf_path)

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
                if not value:
                    continue
                for opt in field["options"]:
                    if opt["option"] == value:
                        x1, y1, x2, y2 = opt["x1"], opt["y1"], opt["x2"], opt["y2"]
                        center_x, center_y = (x1+x2)/2, (y1+y2)/2
                        box_width, box_height = x2-x1, y2-y1
                        tick_size = min(box_width, box_height) * 0.6
                        start1 = (center_x - tick_size*0.4, center_y - tick_size*0.2)
                        mid = (center_x, center_y + tick_size*0.3)
                        end2 = (center_x + tick_size*0.6, center_y - tick_size*0.4)
                        page.draw_line(start1, mid, color=(0,0,0), width=2)
                        page.draw_line(mid, end2, color=(0,0,0), width=2)
                        break
            elif "x1" in field and "x2" in field and "y1" in field and "y2" in field:
                x1, y1, x2, y2 = field["x1"], field["y1"], field["x2"], field["y2"]
                rect_width, rect_height = x2 - x1, y2 - y1
                if field_mode == "charwise":
                    if not value:
                        continue
                    num_boxes = field.get("num_boxes", len(value))
                    if num_boxes == 0:
                        continue
                    char_spacing = rect_width / num_boxes
                    start_x = x1 + char_spacing / 2
                    y = y1 + rect_height / 2 + font_size / 3
                    for i in range(min(len(value), num_boxes)):
                        char = value[i]
                        char_x = start_x + i * char_spacing
                        char_width = fitz.get_text_length(char, fontname=font_family, fontsize=font_size)
                        centered_char_x = char_x - (char_width / 2)
                        page.insert_text((centered_char_x, y), char, fontname=font_family, fontsize=font_size, color=(0,0,0))
                else:
                    if not value:
                        continue
                    text_width = fitz.get_text_length(value, fontname=font_family, fontsize=font_size)
                    x = x1 + (rect_width - text_width) / 2
                    y = y1 + rect_height / 2 + font_size / 3
                    page.insert_text((x, y), value, fontname=font_family, fontsize=font_size, color=(0,0,0))
            else:
                if not value:
                    continue
                x, y = field["x"], field["y"]
                page.insert_text((x, y), value, fontname=font_family, fontsize=font_size, color=(0,0,0))

        hint = row_dict.get("Full Name", row_dict.get("customer_name", "unnamed")).replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{template_id}_{hint}_{timestamp}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        doc.save(output_path)
        doc.close()
        output_paths.append(output_path)
        print(f"Filled and saved: {output_path}")

    wb.close()
    return "|".join(output_paths)


# ----------------------
# Example Usage (Testing)
# ----------------------
if __name__ == "__main__":
    print("Ready for AA360!")
    print("Use map_pdf_template(args) with [pdf_path, optional_template_id, optional_mappings_dir].")
    print("Use fill_pdfs_from_excel(args) with [mappings_dir, excel_path, output_dir, optional_row_index].")

