import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image

# -----------------------------
# Step 1: Image Captioning with BLIP-2
# -----------------------------
print("📸 Generating Image Captions using BLIP-2...")

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)

image_paths = ["D:/OneDrive/Desktop/do2/KakaoTalk_20241230_172527757.jpg", 
               "D:/OneDrive/Desktop/do2/ㅣ뮤.jpg"]
captions = []

for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    captions.append(caption)
    print(f"📝 Caption for {img_path}: {caption}")

# -----------------------------
# Step 2: Text Matching with OpenCLIP
# -----------------------------
print("\n✍️ Processing User Input and Captions...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# User Input
user_input = input("\n📝 Please type a description or prompt for the images: ")
text_inputs = clip_processor(text=[user_input], return_tensors="pt", padding=True, truncation=True)
text_features = clip_model.get_text_features(**text_inputs)
text_features = F.normalize(text_features, dim=-1)

# Encode Captions
caption_features = []
for caption in captions:
    inputs = clip_processor(text=[caption], return_tensors="pt", padding=True, truncation=True)
    caption_embedding = clip_model.get_text_features(**inputs)
    caption_embedding = F.normalize(caption_embedding, dim=-1)
    caption_features.append(caption_embedding)

# Calculate Similarity
print("\n🔍 Calculating Similarity Scores...")
similarities = [torch.matmul(text_features, cap_feat.T).item() for cap_feat in caption_features]
best_match_idx = similarities.index(max(similarities))
best_score = max(similarities)

# Result Display
confidence_threshold = 0.7
if best_score < confidence_threshold:
    print("\n🤔 I couldn't confidently match any image to the description provided.")
else:
    print(f"\n✅ Best Match Found:")
    print(f" - Caption: {captions[best_match_idx]}")
    print(f" - Image: {image_paths[best_match_idx]}")
    print(f" - Confidence Score: {best_score:.4f}")
