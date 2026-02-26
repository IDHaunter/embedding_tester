import argparse
from pathlib import Path
from markitdown import MarkItDown


def convert_email(input_path: Path, output_path: Path | None = None, preview: bool = False):
    """
    Конвертирует сообщения Outlook (.msg) и электронную почту (.eml) в Markdown
    """

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return

    # Добавляем поддержку .msg и .eml
    supported_extensions = [".msg", ".eml"]
    if input_path.suffix.lower() not in supported_extensions:
        print(f"[ERROR] Unsupported extension: {input_path.suffix}. Expected: {supported_extensions}")
        return

    try:
        # Инициализируем MarkItDown
        md = MarkItDown()
        result = md.convert(str(input_path))

        markdown_text = result.text_content

        # Сохранение в файл
        if output_path:
            output_path.write_text(markdown_text, encoding="utf-8")
            print(f"[OK] Markdown saved to: {output_path}")

        # Предварительный просмотр в консоли
        if preview:
            print("\n" + "="*20 + " EMAIL MARKDOWN PREVIEW " + "="*20 + "\n")
            # Выводим первые 2000 символов, чтобы не забивать лог
            print(markdown_text[:2000])
            if len(markdown_text) > 2000:
                print("\n... [content truncated] ...")
            print("\n" + "="*25 + " END PREVIEW " + "="*25 + "\n")

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        print("[TIP] Make sure you have installed: pip install \"markitdown[outlook]\"")


def main():
    parser = argparse.ArgumentParser(description="Outlook (.msg/.eml) → Markdown tester using MarkItDown")
    parser.add_argument("input", help="Path to Email file (.msg or .eml)")
    parser.add_argument("-o", "--output", help="Output markdown file (.md)")
    parser.add_argument("--preview", action="store_true", help="Print preview to console")

    args = parser.parse_args()

    input_path = Path(args.input)

    # Если выходной путь не указан, создаем .md файл рядом с оригиналом
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    convert_email(input_path, output_path, args.preview)


if __name__ == "__main__":
    main()

'''
    EXAMPLE OF USAGE:
        python parsing_outlook_md.py test_outlook.msg
        python parsing_outlook_md.py test_outlook.msg -o result.md
        python parsing_outlook_md.py test_outlook.eml --preview
'''