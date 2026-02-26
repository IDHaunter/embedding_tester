import argparse
from pathlib import Path
from markitdown import MarkItDown


def convert_file(input_path: Path, output_path: Path | None = None, preview: bool = False):
    """
    Конвертирует Excel (.xlsx / .xltm) в Markdown
    """

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    if input_path.suffix.lower() not in [".xlsx", ".xltm"]:
        print(f"[ERROR] Unsupported extension: {input_path.suffix}")
        return

    try:
        md = MarkItDown(enable_plugins=False)
        result = md.convert(str(input_path))

        markdown_text = result.text_content

        if output_path:
            output_path.write_text(markdown_text, encoding="utf-8")
            print(f"[OK] Markdown saved to: {output_path}")

        if preview:
            print("\n===== MARKDOWN PREVIEW =====\n")
            print(markdown_text[:2000])  # ограничим вывод
            print("\n===== END PREVIEW =====\n")

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Excel (.xlsx/.xltm) → Markdown tester using MarkItDown")
    parser.add_argument("input", help="Path to Excel file (.xlsx or .xltm)")
    parser.add_argument("-o", "--output", help="Output markdown file (.md)")
    parser.add_argument("--preview", action="store_true", help="Print preview to console")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    convert_file(input_path, output_path, args.preview)


if __name__ == "__main__":
    main()

'''
    EXAMPLE OF USAGE:
        python parsing_excel_md.py test3.xltm
        python parsing_excel_md.py test3.xltm -o result.md
        python parsing_excel_md.py test3.xltm --preview
'''