import json
from pathlib import Path
import pdfplumber

def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Parses a single PDF file page by page, extracting raw text 
    and capturing structural document metadata.
    """
    pages = []
    full_text = ""
    
    print(f"🔄 Processing: {pdf_path.name}")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages.append({
                    "page": page_num,
                    "text": page_text
                })
                # Append page text with a structural boundary marker
                full_text += page_text + f"\n\n--- PAGE {page_num} ---\n\n"
        
        metadata = {
            "source": pdf_path.name,
            "total_pages": len(pdf.pages)
        }
        
    return {
        "text": full_text.strip(),
        "pages": pages,
        "metadata": metadata
    }

def main():
    # Define and resolve absolute paths relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]  # Steps back up to solar_explorar/
    
    raw_dir = project_root / "data" / "raw"
    extracted_dir = project_root / "data" / "extracted"
    
    # Ensure the output directory exists before writing files
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab all PDFs in the raw data folder
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {raw_dir}. Please drop your files there first!")
        return

    print(f"📂 Found {len(pdf_files)} PDFs to process.\n---")
    
    manifest = {}

    for pdf_path in pdf_files:
        try:
            # 1. Parse the PDF layout
            extracted_data = extract_text_from_pdf(pdf_path)
            
            # 2. Define the exact output text file path
            txt_filename = f"{pdf_path.stem}.txt"
            txt_output_path = extracted_dir / txt_filename
            
            # 3. Save the plaintext content
            with open(txt_output_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(extracted_data["text"])
            
            # 4. Save metadata track to a master manifest dictionary
            manifest[pdf_path.name] = extracted_data["metadata"]
            print(f"✅ Successfully saved: {txt_filename}")
            
        except Exception as e:
            print(f"❌ Failed to extract {pdf_path.name}. Error: {str(e)}")

    # 5. Write a master metadata file tracking your total chunk foundation
    metadata_json_path = extracted_dir / "metadata_manifest.json"
    with open(metadata_json_path, "w", encoding="utf-8") as json_file:
        json.dump(manifest, json_file, indent=4)
        
    print(f"\n✨ Extraction complete! Files saved to {extracted_dir}")
    print(f"📄 Metadata tracking manifest updated at: {metadata_json_path.name}")

if __name__ == "__main__":
    main()