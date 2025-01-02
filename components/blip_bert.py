from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

if __name__ == "__main__":
    # === Load Models === #
    # CLIP for image-text matching
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # BLIP for image captioning
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # SBERT for semantic similarity
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # GPT-2 for refining user prompts
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model.eval()

    # === Load Images === #
    pic = [
        r"C:\Users\User\Desktop\do2\charger.webp",
        r"C:\Users\User\Desktop\do2\cat.jpg",
        r"C:\Users\User\Desktop\do2\apple.webp",
        r"C:\Users\User\Desktop\do2\KakaoTalk_20241230_172527757.jpg",
        r"C:\Users\User\Desktop\do2\ㅣ뮤.jpg"
    ]
    images = [Image.open(img_path).convert("RGB") for img_path in pic]

    # === Generate Captions with BLIP === #
    image_descriptions = []
    for idx, image in enumerate(images):
        blip_inputs = blip_processor(images=image, return_tensors="pt")
        blip_outputs = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
        image_descriptions.append(caption)
        print(f"🖼️ Image {idx + 1}: {caption}")

    # === User Input === #
    user_description = input("\n📝 Please type a description or prompt for the images: ")

    # === Refine User Input with GPT-2 === #
    gpt2_input_text = f"Refine this prompt for an image: '{user_description}'"
    gpt2_inputs = gpt2_tokenizer.encode(gpt2_input_text, return_tensors="pt")
    gpt2_output = gpt2_model.generate(
        gpt2_inputs,
        max_length=20,
        num_return_sequences=1,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        do_sample=True,
        top_k=50
    )
    refined_description = gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True)
    print(f"\n🔄 Refined User Input: {refined_description}")

    # === Compute Semantic Similarity with SBERT === #
    print("\n🔄 Comparing refined description with image captions using SBERT...")

    # Encode user description and image captions with SBERT
    user_embedding = sbert_model.encode(refined_description, convert_to_tensor=True)
    caption_embeddings = sbert_model.encode(image_descriptions, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = util.pytorch_cos_sim(user_embedding, caption_embeddings)
    best_match_idx = similarities.argmax().item()
    best_similarity_score = similarities[0][best_match_idx].item()

    # Print Debugging Information for Similarity Scores
    print("\n📊 Similarity Scores:")
    for idx, score in enumerate(similarities[0]):
        print(f"🖼️ Image {idx + 1}: {image_descriptions[idx]} (Similarity: {score:.2f})")

    # === Evaluate Match === #
    confidence_threshold = 0.3  # Reduced threshold for flexibility

    if best_similarity_score < confidence_threshold:
        print("\n🤔 I couldn't confidently match any image to the description provided.")
    else:
        best_description = image_descriptions[best_match_idx]
        print(f"\n✅ The image that best matches your description '{refined_description}' is: {best_description}")
        print(f"🖼️ Image {best_match_idx + 1} (Semantic Similarity: {best_similarity_score:.2f})")
