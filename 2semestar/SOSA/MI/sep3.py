import os
from PyPDF2 import PdfReader, PdfWriter

output_writer = PdfWriter()

# Loop through all PDF files in the current directory
for filename in os.listdir('.'):
    if filename.lower().endswith('.pdf'):
        try:
            reader = PdfReader(filename)
            if len(reader.pages) >= 3:
                output_writer.add_page(reader.pages[2])  # 3rd page (index 2)
                print(f"Added 3rd page from {filename}")
            else:
                print(f"{filename} has less than 3 pages. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save the combined output if we got at least one page
if len(output_writer.pages) > 0:
    with open("combined_page3s.pdf", "wb") as f:
        output_writer.write(f)
    print("Saved combined 3rd pages to combined_page3s.pdf")
else:
    print("No 3rd pages were found in any PDFs.")
