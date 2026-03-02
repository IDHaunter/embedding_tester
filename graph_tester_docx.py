import argparse
from pathlib import Path
from markitdown import MarkItDown

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)


def convert_docx_to_markdown(input_path: Path) -> str:
    md = MarkItDown(enable_plugins=False)
    result = md.convert(str(input_path))
    return result.text_content


def hierarchical_split(markdown_text: str, chunk_size=800, chunk_overlap=150):
    """
    Hierarchical document splitting pipeline.

    This function performs a two-level split:

    LEVEL 1: Split the Markdown document by headers (H1–H4)
             using MarkdownHeaderTextSplitter.
             This preserves document structure (sections/subsections).

    LEVEL 2: Split each section into smaller overlapping chunks
             using RecursiveCharacterTextSplitter.
             This ensures chunks fit into LLM context windows.

    The result is a flat list of chunks, but each chunk retains
    structural metadata (section index and section path),
    which can later be used to build graph edges such as:
        - sequential edges
        - same-section edges
        - hierarchical edges

    Parameters:
        markdown_text (str): Full Markdown document content.
        chunk_size (int): Maximum size of each chunk (in characters).
        chunk_overlap (int): Overlap between chunks to preserve context continuity.

    Returns:
        list[dict]: List of chunk dictionaries with structural metadata.
    """

    # Final result container
    all_chunks = []

    # -----------------------------------------------------------
    # STEP 1 — Define which Markdown headers represent hierarchy
    #
    # We explicitly define which header levels we want to split on.
    # Each tuple maps Markdown syntax ("#") to a logical name ("H1").
    #
    # This allows us to preserve document hierarchy:
    #   H1 > H2 > H3 > H4
    #
    # Later, this hierarchy can be used to build:
    #   - parent-child relationships
    #   - section-based graph connections
    # -----------------------------------------------------------
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]

    # Create a header-aware splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    # -----------------------------------------------------------
    # STEP 2 — Perform structural split
    #
    # This produces a list of "sections".
    #
    # Each section is a Document object containing:
    #   - page_content  → text inside that section
    #   - metadata      → dictionary with header hierarchy
    #
    # Example metadata:
    #   {
    #       "H1": "Installation",
    #       "H2": "Database Setup"
    #   }
    #
    # This metadata is critical for hierarchical RAG.
    # -----------------------------------------------------------
    sections = markdown_splitter.split_text(markdown_text)

    # -----------------------------------------------------------
    # STEP 3 — Create a character-based chunk splitter
    #
    # RecursiveCharacterTextSplitter:
    #   - respects sentence/paragraph boundaries when possible
    #   - enforces max chunk_size
    #   - adds chunk_overlap to preserve context continuity
    #
    # Overlap is very important for:
    #   - LLM reasoning continuity
    #   - preventing answer loss at chunk boundaries
    # -----------------------------------------------------------
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # -----------------------------------------------------------
    # STEP 4 — Iterate over structured sections
    #
    # section_index represents a structural block defined by headers.
    # -----------------------------------------------------------
    for section_index, section in enumerate(sections):

        # -------------------------------------------------------
        # Build a readable hierarchical path for debugging
        #
        # Example:
        #   "Installation > Database Setup > PostgreSQL"
        #
        # This will later allow:
        #   - same-section graph edges
        #   - section-aware retrieval
        # -------------------------------------------------------
        section_path = " > ".join(
            [v for v in section.metadata.values()]
        )

        # -------------------------------------------------------
        # STEP 5 — Split section content into LLM-sized chunks
        #
        # We now move from structure-aware splitting
        # to size-aware splitting.
        #
        # Important:
        #   Structural boundaries (headers) are preserved,
        #   but within each section we allow chunking.
        # -------------------------------------------------------
        sub_chunks = chunk_splitter.split_text(section.page_content)

        # -------------------------------------------------------
        # STEP 6 — Create final chunk records
        #
        # Each chunk keeps:
        #   - global chunk_id
        #   - section_index (structural parent)
        #   - section_path (hierarchical context)
        #   - raw text
        #
        # Later you can extend this with:
        #   - prev_chunk_id
        #   - next_chunk_id
        #   - block_type
        #   - token_length
        #   - embedding
        # -------------------------------------------------------

         # Global chunk counter (unique ID inside document)
        chunk_id = 0

        for sub_index, chunk in enumerate(sub_chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "section_index": section_index,
                "section_path": section_path,
                "text": chunk
            })

            chunk_id += 1

    # -----------------------------------------------------------
    # Final result:
    # Flat list of chunks with preserved hierarchy metadata.
    #
    # This structure is ready for:
    #   - Vector indexing
    #   - BM25 indexing
    #   - Graph edge construction
    # -----------------------------------------------------------
    return all_chunks


def save_chunks(output_path: Path, chunks):
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK_ID: {chunk['chunk_id']}\n")
            f.write(f"SECTION_INDEX: {chunk['section_index']}\n")
            f.write(f"SECTION_PATH: {chunk['section_path']}\n")
            f.write("-" * 80 + "\n")
            f.write(chunk["text"].strip() + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="DOCX → Markdown → Hierarchical Chunk Tester"
    )
    parser.add_argument("input", help="Path to DOCX file")
    parser.add_argument("-o", "--output", help="Output txt file")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print("[ERROR] File not found")
        return

    if input_path.suffix.lower() != ".docx":
        print("[ERROR] Only .docx supported")
        return

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_suffix(".chunks.txt")
    )

    print("[INFO] Converting DOCX to Markdown...")
    markdown_text = convert_docx_to_markdown(input_path)

    print("[INFO] Splitting hierarchically...")
    chunks = hierarchical_split(markdown_text)

    print(f"[INFO] Total chunks: {len(chunks)}")

    save_chunks(output_path, chunks)

    print(f"[OK] Saved to: {output_path}")

    if args.preview:
        print("\n===== PREVIEW =====\n")
        for chunk in chunks[:3]:
            print(f"[{chunk['chunk_id']}] {chunk['section_path']}")
            print(chunk["text"][:300])
            print("\n")


if __name__ == "__main__":
    main()


"""
USAGE:
    python graph_tester_docx.py file.docx
    python graph_tester_docx.py file.docx --preview
    python graph_tester_docx.py file.docx -o result.txt
"""