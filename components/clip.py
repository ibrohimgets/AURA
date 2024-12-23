from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

if __name__ == "__main__":
    # ✅ Load CLIP Model and Processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ✅ Load BLIP Model and Processor
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # ✅ Load Images
    pics = [
        r"C:\Users\User\Desktop\do2\banana.jpg",
        r"C:\Users\User\Desktop\do2\Artboard1_5f868df7-e28d-4e43-a42d-62be1fdb81b7.webp"
    ]
    images = [Image.open(img_path).convert("RGB") for img_path in pics]

    # ✅ Generate Descriptions for Each Image Using BLIP
    image_descriptions = []
    for idx, image in enumerate(images):
        blip_inputs = blip_processor(images=image, return_tensors="pt")
        blip_outputs = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
        image_descriptions.append(caption)
        print(f"📝 Image {idx + 1}: {caption} (Path: {pics[idx]})")

    # ✅ Multi-Prompt Text Descriptions for Better Matching
    descriptions = [
        "I need something to charge my phone."
    ]

    # ✅ Process Images and Text with CLIP
    clip_inputs = clip_processor(
        text=descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # ✅ Get CLIP Predictions
    outputs = clip_model(**clip_inputs)
    logits_per_image = outputs.logits_per_image  # Shape: (num_text_descriptions, num_images)
    probs = logits_per_image.softmax(dim=1)  # Probabilities per text across images

    # ✅ Find the Best Match Across Text and Images
    best_text_idx, best_image_idx = divmod(probs.argmax().item(), probs.size(1))
    best_confidence = probs[best_text_idx][best_image_idx].item()

    # ✅ Safe Index Access
    if best_image_idx < len(image_descriptions):
        best_description = image_descriptions[best_image_idx]
    else:
        best_description = "Unknown Object"

    # ✅ Apply Confidence Threshold & Relative Confidence
    confidence_threshold = 0.7
    second_best_confidence = torch.topk(probs.flatten(), 2).values[-1].item()
    confidence_gap = best_confidence - second_best_confidence

    if (
        best_confidence < confidence_threshold
        or confidence_gap < 0.1
        or descriptions[best_text_idx] == "None of these match."
    ):
        print("\n🤔 I couldn't confidently match any image to the description provided.")
    else:
        print(f"\n✅ The image that best matches the description '{descriptions[best_text_idx]}' is:")
        print(f"🖼️ Image {best_image_idx + 1} (Confidence: {best_confidence:.2f})")
        print(f"📝 Description: {best_description}")
        print(f"📍 Path: {pics[best_image_idx]}")
