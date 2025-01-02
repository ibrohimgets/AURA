import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and preprocess an image
image_path = r"C:\Users\User\Desktop\do2\cat.jpg"
image = Image.open(image_path)

# Process the image and text prompt
inputs = processor(
    text=["A red apple", "A yellow banana"],
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

# Extract features
with torch.no_grad():
    outputs = model(**inputs)

# Extract image and text embeddings
image_features = outputs.image_embeds
text_features = outputs.text_embeds

# Normalize embeddings
image_features = torch.nn.functional.normalize(image_features, dim=-1)
text_features = torch.nn.functional.normalize(text_features, dim=-1)

# Compute similarity
similarity = (image_features @ text_features.T).softmax(dim=-1)

print("Similarity Scores:", similarity)
