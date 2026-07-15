
import os
import subprocess
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    templates_dir = os.path.join(project_dir, "templates")
    map_tool = os.path.join(script_dir, "map_template.py")

    if not os.path.exists(templates_dir):
        print(f"Error: Templates directory not found: {templates_dir}")
        return

    # Get all PDF files in templates/
    pdf_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in: {templates_dir}")
        return

    print(f"Found {len(pdf_files)} PDF template(s) to map:")
    for i, f in enumerate(pdf_files, 1):
        print(f"  {i}. {f}")
    print()

    # Process each template
    for pdf_file in pdf_files:
        template_id = os.path.splitext(pdf_file)[0].lower().replace(" ", "_")
        pdf_path = os.path.join(templates_dir, pdf_file)
        print(f"Processing template: {pdf_file} (ID: {template_id})")
        print(f"Running mapping tool for this template...")
        print("When you're done, click 'Save & Exit' in the GUI to proceed to the next template.")
        print("-" * 60)

        # Run map_template.py
        try:
            subprocess.run([sys.executable, map_tool, "--pdf", pdf_path, "--id", template_id], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error mapping {pdf_file}: {e}")
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

        print()

    print("Done!")

if __name__ == "__main__":
    main()
