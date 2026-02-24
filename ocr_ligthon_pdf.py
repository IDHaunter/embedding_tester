import base64
import requests
import io
import sys
from pdf2image import convert_from_path
from PIL import Image
from timing_decorator import measure_time

ENDPOINT = "http://49.13.101.190:8000/v1/chat/completions"
MODEL = "/lightonai/LightOnOCR-2-1B"


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@measure_time
def ocr_image_markdown(image_base64: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract ALL text from the image and return it as clean Markdown. "
                                "Preserve headings, lists, tables and formatting."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    response = requests.post(ENDPOINT, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

@measure_time
def ocr_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)

    print(f"Total pages: {len(pages)}")
    print("=" * 60)

    for i, page in enumerate(pages, start=1):
        print(f"\n📄 Processing page {i}...\n")

        image_base64 = image_to_base64(page)
        markdown_text = ocr_image_markdown(image_base64)

        print(f"\n----- PAGE {i} -----\n")
        print(markdown_text)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_pdf.py path_to_file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    ocr_pdf(pdf_path)