from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

if __name__ == "__main__":
    # Bring the CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Bring the BLIP
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load images
    pic = [
        r"D:\OneDrive\Desktop\do2\charger.webp",
        r"D:\OneDrive\Desktop\do2\cat.jpg",
        r"D:\OneDrive\Desktop\do2\apple.webp",
        r"D:\OneDrive\Desktop\do2\KakaoTalk_20241230_172527757.jpg",
        r"D:\OneDrive\Desktop\do2\ㅣ뮤.jpg"
    ]
    images = [Image.open(img_path).convert("RGB") for img_path in pic]

    # Generate image captions using BLIP
    image_descriptions = []
    for idx, image in enumerate(images):
        blip_inputs = blip_processor(images=image, return_tensors="pt")
        blip_outputs = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
        image_descriptions.append(caption)
        print(f"Image {idx + 1}: {caption}")

    # User input description
    description = input("\n📝 Please type a description or prompt for the images: ")

    # Additional fallback descriptions
    fallback_descriptions = [
        description,
        "an electronic device",
        "a screen device",
        "Not found!"
    ]

    # Prepare CLIP inputs
    clip_inputs = clip_processor(
        text=fallback_descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Perform image-text matching with CLIP
    outputs = clip_model(**clip_inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Find the best match
    best_flat_idx = probs.argmax().item()
    image_idx, description_idx = divmod(best_flat_idx, probs.shape[1])

    best_confidence = probs[image_idx][description_idx].item()
    confidence_threshold = 0.8  # Slightly increased for stricter matching

    if best_confidence < confidence_threshold:
        print("\n🤔 I couldn't confidently match any image to the description provided. Please refine your input.")
    else:
        best_description = image_descriptions[image_idx] if image_idx < len(image_descriptions) else "Unknown Object"
        print(f"\nThe image that best matches your description '{fallback_descriptions[description_idx]}' is: {best_description}")
        print(f"Image {image_idx + 1} (Confidence: {best_confidence:.2f})")
