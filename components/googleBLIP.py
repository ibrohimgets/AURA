from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import google.generativeai as genai
import atexit
import grpc

# Gemini API
genai.configure(api_key="AIzaSyDbrHH0Takjs3gKdpWXzkce2psfsIUOZLw")
model = genai.GenerativeModel("gemini-1.5-flash")

if __name__ == "__main__":
    # BLIP model
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # load the pics
    pic = [
        r"D:\OneDrive\Desktop\do2\charger.webp",
        r"D:\OneDrive\Desktop\do2\cat.jpg",
        r"D:\OneDrive\Desktop\do2\apple.webp",
        r"D:\OneDrive\Desktop\do2\KakaoTalk_20241230_172527757.jpg",
        r"D:\OneDrive\Desktop\do2\ㅣ뮤.jpg",
    ]
    images = [Image.open(img_path).convert("RGB") for img_path in pic]

    # pic captions 
    image_descriptions = []
    for idx, image in enumerate(images):
        blip_inputs = blip_processor(images=image, return_tensors="pt")
        blip_outputs = blip_model.generate(**blip_inputs)
        caption = blip_processor.decode(blip_outputs[0], skip_special_tokens=True)
        image_descriptions.append(caption)
        print(f"Image {idx + 1}: {caption}")

    # user input
    user_description = input("Please enter the prompt:")

    # Gemini prompt
    prompt = f"""
    I have here some pictures. Please match the user's description to the most relevant caption.
    User Description: "{user_description}"
    Captions:
    {', '.join([f'{idx+1}. {desc}' for idx, desc in enumerate(image_descriptions)])}
    Return the number of the most relevant caption and explain your reasoning.
    """

    try:
        response = model.generate_content(prompt)
        gemini_output = response.text
        print("Result!")
        print(gemini_output)

        # Extract the matching caption index
        import re
        match = re.search(r'\b(\d+)\b', gemini_output)
        if match:
            best_idx = int(match.group(1)) - 1
            if 0 <= best_idx < len(image_descriptions):
                print(f"\n✅ The image that best matches your description is: {image_descriptions[best_idx]}")
                print(f"Image {best_idx + 1}")
            else:
                print("\n❌ Gemini's response did not provide a valid index.")
        else:
            print("\n❌ Could not extract a valid match from Gemini's response.")

    except Exception as e:
        print(f"\n❌ An error occurred with Gemini API: {e}")
def shutdown_grpc():
    try:
        grpc._channel._Rendezvous.close()
    except Exception:
        pass