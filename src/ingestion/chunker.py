import json
from pathlib import Path
import re

def chunk_text_by_page(page_text: str, source_name: str, page_num: int, target_size: int = 700, overlap: int = 80) -> list:
    """
    Chunks text strictly within the boundaries of a single page,
    ensuring page numbers are accurately retained per text block.
    """
    chunks = []
    start = 0
    text_length = len(page_text)

    while start < text_length:
        end = start + target_size
        chunk = page_text[start:end]

        if end < text_length:
            last_period = chunk.rfind('.')
            if last_period > target_size * 0.7:
                end = start + last_period + 1
                chunk = page_text[start:end]

        cleaned_text = chunk.strip()
        if cleaned_text:
            chunks.append({
                "chunk_id": None,  # Handled downstream sequentially
                "source": source_name,
                "page": page_num,   # Tracked directly from extraction mapping
                "text": cleaned_text
            })

        start = end - overlap

    return chunks

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    
    extracted_dir = project_root / "data" / "extracted"
    output_dir = project_root / "data" / "chunks"
    output_file = output_dir / "chunks.json"
    
    if not extracted_dir.exists():
        print(f"❌ Extracted directory missing. Run extract.py first!")
        return

    txt_files = list(extracted_dir.glob("*.txt"))
    print(f"✂️  Parsing page-aware blocks across {len(txt_files)} extracted files...")
    
    all_chunks = []
    global_chunk_counter = 0

    for txt_path in txt_files:
        original_pdf_name = f"{txt_path.stem}.pdf"
        
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Regex split matching our custom structural layout markers: --- PAGE X ---
        # This divides the text back into individual pages while keeping the page numbers
        page_segments = re.split(r'--- PAGE (\d+) ---', content)
        
        # re.split results layout: [pre-text, "1", page_1_text, "2", page_2_text, ...]
        # We start loop from index 1 to grab the matched digits and text pairs
        for idx in range(1, len(page_segments), 2):
            page_num = int(page_segments[idx])
            page_text = page_segments[idx + 1].strip()
            
            if not page_text:
                continue
                
            # Generate chunks specifically locked to this page identity
            page_chunks = chunk_text_by_page(
                page_text=page_text,
                source_name=original_pdf_name,
                page_num=page_num
            )
            
            # Map sequential ID index over the final global matrix arrays
            for chunk in page_chunks:
                chunk["chunk_id"] = global_chunk_counter
                all_chunks.append(chunk)
                global_chunk_counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(all_chunks, out_f, indent=2, ensure_ascii=False)

    print(f"✅ Page-aware chunking complete. Saved {len(all_chunks)} elements to chunks.json.")

if __name__ == "__main__":
    main()