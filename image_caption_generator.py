#`image_caption_generator.py`

# AI Image Caption Generator

# Author: Preet Kadam
# 3rd Year Project - CSE-AI

import sys
import os
import requests

try:
    import torch
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    from PIL import Image
except ImportError as e:
    missing = str(e).split()[-1].replace("'", "")
    sys.exit(f"❌ Required module '{missing}' is missing. Install all with: pip install torch transformers pillow requests")

# === Load Pretrained Model ===
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    test_image = "test.jpg"
    if not os.path.exists(test_image):
        try:
            url = "https://images.unsplash.com/photo-1602524201271-d3909c9a7b83"
            img = Image.open(requests.get(url, stream=True).raw)
            img.save(test_image)
            print("✅ Downloaded sample image.")
        except Exception as e:
            sys.exit(f"❌ Could not download test image: {e}")

    print("\n🖼️ Generating caption for:", test_image)
    try:
        caption = generate_caption(test_image)
        print("\n📝 Caption:", caption)
    except Exception as e:
        print("❌ Caption generation failed:", e)
