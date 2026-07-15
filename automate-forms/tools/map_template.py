
import argparse
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import fitz

def main():
    parser = argparse.ArgumentParser(description="Map a PDF template by clicking field locations.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF template")
    parser.add_argument("--id", required=True, help="Template ID (used for mapping file name)")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found: {args.pdf}")
        return

    doc = fitz.open(args.pdf)
    page_count = len(doc)
    if page_count == 0:
        print("Error: PDF has no pages")
        doc.close()
        return

    # Get page size from first page (assuming all pages same size)
    first_page = doc[0]
    page_size_pts = [first_page.rect.width, first_page.rect.height]

    fields = []
    undo_stack = []  # To store field history for undo
    redo_stack = []  # To store field history for redo

    # GUI Setup
    root = tk.Tk()
    root.title(f"Map Template: {args.id}")
    root.geometry("1200x850")

    current_page = 0
    zoom = 1.0
    photo = None
    img_width = 0
    img_height = 0

    # Drag variables
    drag_start = None
    current_drag_rect = None
    current_small_boxes = []
    current_option_boxes = []  # For building a multi-option field
    current_field_name = None  # For multi-option field building

    # --- Instructions Frame ---
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

    # Frame for PDF display with scrollbars
    pdf_frame = ttk.Frame(root, padding="10")
    pdf_frame.pack(fill=tk.BOTH, expand=True)

    # Create scrollbars
    v_scrollbar = ttk.Scrollbar(pdf_frame, orient=tk.VERTICAL)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar = ttk.Scrollbar(pdf_frame, orient=tk.HORIZONTAL)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    # Canvas for PDF page
    canvas = tk.Canvas(pdf_frame, bg="white",
                       xscrollcommand=h_scrollbar.set,
                       yscrollcommand=v_scrollbar.set)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Link scrollbars to canvas
    v_scrollbar.config(command=canvas.yview)
    h_scrollbar.config(command=canvas.xview)

    # Frame for controls
    control_frame = ttk.Frame(root, padding="10")
    control_frame.pack(fill=tk.X)

    # Row 0: Page navigation, Zoom, Save & Exit
    # Page navigation
    ttk.Label(control_frame, text="Page:").grid(row=0, column=0, padx=5, sticky=tk.W)
    page_var = tk.StringVar(value=f"{current_page + 1}/{page_count}")
    ttk.Label(control_frame, textvariable=page_var).grid(row=0, column=1, padx=5, sticky=tk.W)
    ttk.Button(control_frame, text="Previous", command=lambda: change_page(-1)).grid(row=0, column=2, padx=5, sticky=tk.W)
    ttk.Button(control_frame, text="Next", command=lambda: change_page(1)).grid(row=0, column=3, padx=5, sticky=tk.W)

    # Zoom controls
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

    # Save & Exit button (RIGHT)
    save_btn = ttk.Button(control_frame, text="✅ Save & Exit", command=lambda: save_and_exit())
    save_btn.grid(row=0, column=10, pady=5, padx=(20, 5), sticky=tk.E)
    control_frame.columnconfigure(10, weight=1)  # Make this column expand to push save button right

    # Row 1: Field name, Font family, Font size
    ttk.Label(control_frame, text="Field Name (exact Excel column):").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
    field_name_var = tk.StringVar()
    field_name_entry = ttk.Entry(control_frame, textvariable=field_name_var, width=25)
    field_name_entry.grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W, pady=5)
    field_name_entry.focus()

    ttk.Label(control_frame, text="Font:").grid(row=1, column=3, padx=5, sticky=tk.W, pady=5)
    font_family_var = tk.StringVar(value="helv")
    font_options = [
        "helv", "helv-oblique", "helv-bold", 
        "cour", "cour-oblique", "cour-bold", 
        "tiro", "tiro-italic", "tiro-bold"
    ]
    font_menu = ttk.Combobox(control_frame, textvariable=font_family_var, values=font_options, state="readonly", width=15)
    font_menu.grid(row=1, column=4, columnspan=2, padx=5, sticky=tk.W, pady=5)

    ttk.Label(control_frame, text="Font Size:").grid(row=1, column=6, padx=5, sticky=tk.W, pady=5)
    font_size_var = tk.IntVar(value=11)
    ttk.Spinbox(control_frame, from_=5, to=20, textvariable=font_size_var, width=5).grid(row=1, column=7, padx=5, sticky=tk.W, pady=5)
    
    # Undo/Redo buttons
    def undo_last():
        if not fields:
            messagebox.showinfo("Info", "Nothing to undo!")
            return
        last_field = fields.pop()
        undo_stack.append(last_field)
        redo_stack.clear()  # Clear redo stack when new undo is done
        update_fields_list()
        render_page()
    ttk.Button(control_frame, text="Undo", command=lambda: undo_last()).grid(row=1, column=8, padx=5, sticky=tk.W, pady=5)
    
    def redo_last():
        if not undo_stack:
            messagebox.showinfo("Info", "Nothing to redo!")
            return
        last_undone = undo_stack.pop()
        fields.append(last_undone)
        update_fields_list()
        render_page()
    ttk.Button(control_frame, text="Redo", command=lambda: redo_last()).grid(row=1, column=9, padx=5, sticky=tk.W, pady=5)

    # Row 2: Mode selection
    mode_var = tk.StringVar(value="normal")
    mode_frame = ttk.Frame(control_frame)
    mode_frame.grid(row=2, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Radiobutton(mode_frame, text="Normal", variable=mode_var, value="normal").pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(mode_frame, text="Character-wise", variable=mode_var, value="charwise").pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(mode_frame, text="Option/Checkbox", variable=mode_var, value="option").pack(side=tk.LEFT, padx=5)
    
    # Row 3: Mode-specific controls
    # Character-wise controls (only shown when mode is charwise)
    num_boxes_frame = ttk.Frame(control_frame)
    num_boxes_frame.grid(row=3, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Label(num_boxes_frame, text="Number of Boxes:").pack(side=tk.LEFT, padx=5)
    num_boxes_var = tk.IntVar(value=10)
    num_boxes_spinbox = ttk.Spinbox(num_boxes_frame, from_=1, to=1000, textvariable=num_boxes_var, width=5, state="normal")
    num_boxes_spinbox.pack(side=tk.LEFT, padx=5)
    
    # Option/Checkbox controls (only shown when mode is option)
    option_frame = ttk.Frame(control_frame)
    option_frame.grid(row=3, column=0, columnspan=10, padx=5, sticky=tk.W, pady=5)
    ttk.Label(option_frame, text="Option Label:").pack(side=tk.LEFT, padx=5)
    option_label_var = tk.StringVar(value="Y")
    option_label_entry = ttk.Entry(option_frame, textvariable=option_label_var, width=10)
    option_label_entry.pack(side=tk.LEFT, padx=5)
    add_option_btn = ttk.Button(option_frame, text="Add Option Box", command=lambda: add_current_as_option())
    add_option_btn.pack(side=tk.LEFT, padx=5)
    finish_options_btn = ttk.Button(option_frame, text="Finish Field", command=lambda: finish_option_field())
    finish_options_btn.pack(side=tk.LEFT, padx=5)
    cancel_options_btn = ttk.Button(option_frame, text="Cancel", command=lambda: cancel_option_field())
    cancel_options_btn.pack(side=tk.LEFT, padx=5)
    ttk.Label(option_frame, text="Options:").pack(side=tk.LEFT, padx=5)
    current_options_label_var = tk.StringVar(value="")
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
    
    toggle_mode_controls()  # Initial state
    mode_var.trace_add("write", lambda *args: toggle_mode_controls())

    # Row 4: Fields list label
    ttk.Label(control_frame, text="Mapped Fields:").grid(row=4, column=0, columnspan=10, sticky=tk.W, pady=(10, 5))

    # Row 5: Fields listbox
    fields_listbox = tk.Listbox(control_frame, height=5)
    fields_listbox.grid(row=5, column=0, columnspan=9, sticky=tk.EW, padx=5, pady=5)
    scrollbar = ttk.Scrollbar(control_frame, orient=tk.VERTICAL, command=fields_listbox.yview)
    scrollbar.grid(row=5, column=9, sticky=tk.NS)
    fields_listbox.config(yscrollcommand=scrollbar.set)
    control_frame.columnconfigure(8, weight=1)  # Make the listbox column expand

    # Row 6: Delete Selected button
    ttk.Button(control_frame, text="Delete Selected", command=lambda: delete_selected_field()).grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.W)

    # Configure grid weights for responsive layout
    # Column 10 has weight=1 to push Save button to right (already set above)
    # Column 8 has weight=1 to make fields listbox expand (already set above)

    def render_page():
        nonlocal photo, img_width, img_height
        page = doc[current_page]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_width = pix.width
        img_height = pix.height
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
        # Draw existing fields
        for idx, field in enumerate(fields):
            if field["page"] == current_page:
                field_mode = field.get("mode", "normal")
                if field_mode == "option":
                    # Draw each option box
                    for opt_idx, opt in enumerate(field["options"]):
                        x1 = opt["x1"] * zoom
                        y1 = opt["y1"] * zoom
                        x2 = opt["x2"] * zoom
                        y2 = opt["y2"] * zoom
                        canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2, tags=f"field_{idx}")
                        # Draw option label
                        canvas.create_text((x1+x2)/2, (y1+y2)/2, text=opt["option"], fill="darkgreen", font=("Arial", 10, "bold"), tags=f"field_{idx}")
                    # Draw field name above first box
                    first_opt = field["options"][0]
                    fx1 = first_opt["x1"] * zoom
                    fy1 = first_opt["y1"] * zoom
                    canvas.create_text(fx1, fy1-10, text=field["field_name"], fill="darkblue", font=("Arial", 10, "bold"), anchor=tk.W, tags=f"field_{idx}")
                else:
                    # Normal or charwise mode
                    x1 = field["x1"] * zoom
                    y1 = field["y1"] * zoom
                    x2 = field["x2"] * zoom
                    y2 = field["y2"] * zoom
                    canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags=f"field_{idx}")
                    canvas.create_text((x1+x2)/2, y1-10, text=field["field_name"], fill="darkblue", font=("Arial", 10, "bold"), tags=f"field_{idx}")
                    # Draw small boxes for character-wise
                    if field_mode == "charwise" and "num_boxes" in field:
                        num_boxes = field["num_boxes"]
                        rect_width = x2 - x1
                        box_width = rect_width / num_boxes
                        for i in range(num_boxes):
                            small_x1 = x1 + i * box_width
                            small_x2 = small_x1 + box_width
                            canvas.create_rectangle(small_x1, y1, small_x2, y2, outline="orange", width=1, tags=f"field_{idx}")
        
        # Draw current option boxes (in progress)
        for opt_idx, opt in enumerate(current_option_boxes):
            x1 = opt["x1"] * zoom
            y1 = opt["y1"] * zoom
            x2 = opt["x2"] * zoom
            y2 = opt["y2"] * zoom
            outline_color = "blue" if not opt["option"] else "green"
            canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2, dash=(5,5) if not opt["option"] else None)
            if opt["option"]:
                canvas.create_text((x1+x2)/2, (y1+y2)/2, text=opt["option"], fill="darkgreen", font=("Arial", 10, "bold"))
        
        page_var.set(f"{current_page + 1}/{page_count}")

    def change_page(delta):
        nonlocal current_page
        new_page = current_page + delta
        if 0 <= new_page < page_count:
            current_page = new_page
            render_page()

    # --- Drag Functions ---
    def on_mouse_down(event):
        nonlocal drag_start, current_drag_rect, current_small_boxes
        field_name = field_name_var.get().strip()
        if not field_name:
            messagebox.showwarning("⚠️ Missing Field Name", "First type a Field Name!")
            field_name_entry.focus()
            return
        # Get canvas coordinates (adjust for scroll)
        x_start = canvas.canvasx(event.x)
        y_start = canvas.canvasy(event.y)
        drag_start = (x_start, y_start)
        current_small_boxes = []
        # Create initial rect
        current_drag_rect = canvas.create_rectangle(x_start, y_start, x_start, y_start, outline="blue", width=2, dash=(5, 5))

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
        
        # Clear previous small boxes first
        for box in current_small_boxes:
            canvas.delete(box)
        current_small_boxes = []
        
        # Draw small boxes if current mode is charwise
        if mode_var.get() == "charwise":
            num_boxes = num_boxes_var.get()
            rect_width = x2 - x1
            if rect_width > 0 and num_boxes > 0:  # Avoid division by zero
                box_width = rect_width / num_boxes
                for i in range(num_boxes):
                    small_x1 = x1 + i * box_width
                    small_x2 = small_x1 + box_width
                    small_box = canvas.create_rectangle(small_x1, y1, small_x2, y2, outline="orange", width=1, dash=(2, 2))
                    current_small_boxes.append(small_box)

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
        # Update the last box to have the label
        last_box["option"] = option_label
        # Update the options label
        current_options = [box["option"] for box in current_option_boxes]
        current_options_label_var.set(", ".join(current_options))
        # Clear drag state
        nonlocal drag_start, current_drag_rect, current_small_boxes
        drag_start = None
        canvas.delete(current_drag_rect)
        for box in current_small_boxes:
            canvas.delete(box)
        current_drag_rect = None
        current_small_boxes = []
        # Render to show the new option box
        render_page()
        # Clear option label for next
        option_label_var.set("")
        
    def finish_option_field():
        nonlocal current_option_boxes, current_field_name
        if not current_option_boxes:
            messagebox.showwarning("Warning", "No options added!")
            return
        field_name = current_field_name if current_field_name else field_name_var.get().strip()
        if not field_name:
            messagebox.showwarning("Warning", "Enter a field name!")
            return
        # Add the field with options
        field_dict = {
            "field_name": field_name,
            "page": current_page,
            "font_family": font_family_var.get(),
            "font_size": font_size_var.get(),
            "mode": "option",
            "options": current_option_boxes
        }
        fields.append(field_dict)
        redo_stack.clear()  # Clear redo stack when new field is added
        # Reset
        current_option_boxes = []
        current_field_name = None
        current_options_label_var.set("")
        # Update UI
        update_fields_list()
        render_page()
        field_name_var.set("")
        field_name_entry.focus()
        
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
        
    def on_mouse_up(event):
        nonlocal drag_start, current_drag_rect, current_small_boxes, current_option_boxes, current_field_name
        mode = mode_var.get()
        if drag_start is None or current_drag_rect is None:
            return
        x_start, y_start = drag_start
        x_end = canvas.canvasx(event.x)
        y_end = canvas.canvasy(event.y)
        # Get min/max to handle drag in any direction
        x1 = min(x_start, x_end)
        y1 = min(y_start, y_end)
        x2 = max(x_start, x_end)
        y2 = max(y_start, y_end)
        # Convert to PDF points
        x1_pts = x1 / zoom
        y1_pts = y1 / zoom
        x2_pts = x2 / zoom
        y2_pts = y2 / zoom
        
        if mode == "option":
            # Start building option field if first box
            if not current_option_boxes:
                current_field_name = field_name_var.get().strip()
                if not current_field_name:
                    messagebox.showwarning("⚠️ Missing Field Name", "First type a Field Name!")
                    field_name_entry.focus()
                    # Cleanup
                    drag_start = None
                    canvas.delete(current_drag_rect)
                    for box in current_small_boxes:
                        canvas.delete(box)
                    current_drag_rect = None
                    current_small_boxes = []
                    return
            # Add to current_option_boxes (without label yet)
            current_option_boxes.append({
                "x1": x1_pts,
                "y1": y1_pts,
                "x2": x2_pts,
                "y2": y2_pts,
                "option": ""
            })
            # Cleanup drag
            drag_start = None
            canvas.delete(current_drag_rect)
            for box in current_small_boxes:
                canvas.delete(box)
            current_drag_rect = None
            current_small_boxes = []
            # Render to show the new box
            render_page()
            # Focus option label entry
            option_label_entry.focus()
        else:
            # Normal or charwise mode
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
            redo_stack.clear()  # Clear redo stack when new field is added
            # Cleanup
            drag_start = None
            canvas.delete(current_drag_rect)
            for box in current_small_boxes:
                canvas.delete(box)
            current_drag_rect = None
            current_small_boxes = []
            # Update UI
            update_fields_list()
            render_page()
            field_name_var.set("")
            field_name_entry.focus()

    # Bind drag events
    canvas.bind("<Button-1>", on_mouse_down)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_up)

    # Mouse wheel zoom
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

    # Bind Ctrl+Z for undo and Ctrl+Y for redo
    def on_ctrl_z(event):
        undo_last()
    root.bind("<Control-z>", on_ctrl_z)
    
    def on_ctrl_y(event):
        redo_last()
    root.bind("<Control-y>", on_ctrl_y)

    def update_fields_list():
        fields_listbox.delete(0, tk.END)
        for idx, field in enumerate(fields):
            fields_listbox.insert(tk.END, f"{field['field_name']} (Page {field['page'] + 1})")

    def delete_selected_field():
        selected = fields_listbox.curselection()
        if selected:
            idx = selected[0]
            deleted_field = fields.pop(idx)
            undo_stack.append(deleted_field)
            redo_stack.clear()  # Clear redo stack when field is deleted
            update_fields_list()
            render_page()

    def save_and_exit():
        if len(fields) == 0:
            confirm = messagebox.askyesno("⚠️ No Fields Mapped", "You haven't mapped any fields. Save anyway?")
            if not confirm:
                return
        mapping = {
            "template_id": args.id,
            "pdf_path": args.pdf,
            "page_count": page_count,
            "page_size_pts": page_size_pts,
            "fields": fields
        }
        mappings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mappings")
        os.makedirs(mappings_dir, exist_ok=True)
        mapping_file = os.path.join(mappings_dir, f"{args.id}.json")
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)
        messagebox.showinfo("✅ Success!", f"Mapping saved to:\n{mapping_file}\n\nNow you can fill this template!")
        doc.close()
        root.destroy()

    render_page()
    root.mainloop()

if __name__ == "__main__":
    main()
