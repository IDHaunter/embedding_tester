import base64
import requests
import io
from PIL import Image, ImageDraw, ImageFont

ENDPOINT = "http://49.13.101.190:8000/v1/chat/completions"
MODEL = "/lightonai/LightOnOCR-2-1B"

# 1️⃣ Create simple image with "Hello World"
img = Image.new("RGB", (400, 150), color="white")
draw = ImageDraw.Draw(img)

# Use default font
draw.text((50, 50), "Hello World", fill="black")

# 2️⃣ Convert to base64
buffer = io.BytesIO()
img.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

# 3️⃣ Build request
payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract text from the image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    },
                },
            ],
        }
    ],
    "max_tokens": 512,
    "temperature": 0.0,
}
print("Payload:", payload)

# 4️⃣ Send request
response = requests.post(ENDPOINT, json=payload)

print("Status:", response.status_code)
print("Response:")
print(response.json()["choices"][0]["message"]["content"])