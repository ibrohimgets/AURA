from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

if __name__ == "__main__":
    # ✅ Load the CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ✅ Load the BLIP model and processor
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # ✅ Load images
    image_paths = [
        r"C:\Users\User\Desktop\do2\banana.jpg",
        r"C:\Users\User\Desktop\do2\cat.jpg"
    ]
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

    # ✅ Generate descriptions for each image using BLIP
    image_descriptions = []
    for idx, image in enumerate(images):
        blip_inputs = blip_processor(images=image, return_tensors="pt")
        blip_outputs = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
        image_descriptions.append(caption)
        print(f" Image {idx + 1}: {caption} (Path: {image_paths[idx]})")

    # ✅ User-provided text description
    description = "A very fluffy pet!"

    # Add a fallback description
    descriptions = [description, "An unknown object"]

    # ✅ Process the images and text with CLIP
    clip_inputs = clip_processor(
        text=descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # ✅ Get CLIP predictions
    outputs = clip_model(**clip_inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # Compare across images

    # ✅ Find the best match
    best_idx = probs.argmax().item()
    best_confidence = probs[0][best_idx].item()
    best_description = image_descriptions[best_idx] if best_idx < len(image_descriptions) else "Unknown Object"

    # ✅ Set a confidence threshold
    confidence_threshold = 0.7

    if best_confidence < confidence_threshold or best_idx == len(images):
        print("\n🤔 I couldn't confidently match any image to the description provided.")
    else:
        print(f"\n✅ The image that best matches the description '{description}' is: {best_description}")
        print(f"🖼️ Image {best_idx + 1} (Confidence: {best_confidence:.2f})")
        

