import os
import openai
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("❌ OpenAI API key not found! Please set it in your .env file.")

# -----------------------------
# Step 1: Generate Captions with BLIP
# -----------------------------
def generate_captions(image_paths):
    """
    Generate image captions using BLIP.
    """
    print("📸 Generating Image Captions using BLIP...")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    
    captions = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        captions.append(caption)
        print(f"📝 Caption for {img_path}: {caption}")
    return captions

# -----------------------------
# Step 2: Match Captions with GPT
# -----------------------------
def match_with_gpt(user_input, captions):
    """
    Match user input with captions using GPT.
    """
    print("\n🤖 Matching with GPT...")
    
    # Build the GPT Prompt
    prompt = f"""
You are an AI assistant tasked with matching a user's text prompt to the most relevant image caption.

User Prompt: "{user_input}"

Image Captions:
{chr(10).join([f"{i+1}. {caption}" for i, caption in enumerate(captions)])}

Which caption best matches the user's prompt? Please return the number of the best-matching caption and explain your choice.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use 'gpt-3.5-turbo' if GPT-4 is unavailable
            messages=[
                {"role": "system", "content": "You are an assistant specialized in semantic reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2
        )

        # Parse GPT response
        reply = response['choices'][0]['message']['content']
        print("\n💬 GPT Response:", reply)
        
        # Extract the matching caption index
        import re
        match = re.search(r'(\d+)', reply)
        if match:
            best_idx = int(match.group(1)) - 1
            return best_idx
        else:
            print("🤔 GPT did not return a valid caption number.")
            return None

    except openai.OpenAIError as e:
        print(f"❌ OpenAI API Error: {e}")
        return None

# -----------------------------
# Step 3: Main Program
# -----------------------------
if __name__ == '__main__':
    # Define image paths
    image_paths = [
        "D:/OneDrive/Desktop/do2/KakaoTalk_20241230_172527757.jpg",
        "D:/OneDrive/Desktop/do2/ㅣ뮤.jpg"
    ]
    
    # Generate captions using BLIP
    captions = generate_captions(image_paths)
    
    # User Input
    user_input = input("\n📝 Please type a description or prompt for the images: ")
    
    # Match captions using GPT
    best_match_idx = match_with_gpt(user_input, captions)
    if best_match_idx is not None and 0 <= best_match_idx < len(captions):
        print(f"\n✅ Best Match Found:")
        print(f" - Caption: {captions[best_match_idx]}")
        print(f" - Image: {image_paths[best_match_idx]}")
    else:
        print("\n❌ No valid match was returned by GPT.")
