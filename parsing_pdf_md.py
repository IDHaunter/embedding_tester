import argparse
from pathlib import Path
from markitdown import MarkItDown


def convert_pdf(input_path: Path, output_path: Path | None = None, preview: bool = False):
    """
    Конвертирует PDF → Markdown
    """

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    if input_path.suffix.lower() != ".pdf":
        print(f"[ERROR] Unsupported extension: {input_path.suffix}")
        return

    try:
        print("[INFO] Initializing MarkItDown...")
        md = MarkItDown(enable_plugins=False)

        print("[INFO] Converting PDF to Markdown...")
        result = md.convert(str(input_path))

        markdown_text = result.text_content

        if not markdown_text.strip():
            print("[WARNING] Extracted text is empty.")
            return

        if output_path:
            output_path.write_text(markdown_text, encoding="utf-8")
            print(f"[OK] Markdown saved to: {output_path}")

        if preview:
            print("\n===== MARKDOWN PREVIEW =====\n")
            print(markdown_text[:3000])  # ограниченный вывод
            print("\n===== END PREVIEW =====\n")

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="PDF → Markdown tester using MarkItDown")
    parser.add_argument("input", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output markdown file (.md)")
    parser.add_argument("--preview", action="store_true", help="Print preview to console")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    convert_pdf(input_path, output_path, args.preview)


if __name__ == "__main__":
    main()

'''
    EXAMPLE OF USAGE:
        python parsing_pdf_md.py test_order.pdf
        python parsing_pdf_md.py test_order.pdf -o result.md
        python parsing_pdf_md.py test_order.pdf --preview
'''