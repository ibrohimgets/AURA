from transformers import CLIPProcessor, CLIPModel, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torch

if __name__ == "__main__":
    # Clip model here
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load GPT-2 model and tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set GPT-2 to evaluation mode
    gpt2_model.eval()

    # Load your image
    image_path = r"C:\Users\User\Desktop\do2\cat.jpg"
    image = Image.open(image_path).convert("RGB")

    # Contextual Text Descriptions
    descriptions = [
        "A car",
        "A ripe yellow banana",
        "A cute cat sitting on a chair",
        "A playful dog",
        "An unknown object"
    ]

    # Process the image and text with CLIP
    clip_inputs = clip_processor(
        text=descriptions,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Get CLIP predictions
    outputs = clip_model(**clip_inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Confidence Threshold
    confidence_threshold = 0.6

    # Find the best match and its confidence
    best_idx = probs.argmax().item()
    best_confidence = probs[0][best_idx].item()
    best_description = descriptions[best_idx]

    if best_confidence < confidence_threshold or best_description == "An unknown object":
        print("🤔 I don't think there's a proper object that matches your description.")
    else:
        print(f"🖼️ Best Match (CLIP): {best_description} (Confidence: {best_confidence:.2f})")

        # Pass the description to GPT-2 for refinement
        gpt2_input_text = f"The image shows {best_description}. It seems to be"
        gpt2_inputs = gpt2_tokenizer.encode(gpt2_input_text, return_tensors="pt")

        gpt2_output = gpt2_model.generate(
            gpt2_inputs,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode and print GPT-2 response
        gpt2_response = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
        print(f"🤖 Refined Description (GPT-2): {gpt2_response}")
