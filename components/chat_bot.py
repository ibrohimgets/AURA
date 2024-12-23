from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto")

# Move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# User input
user_input = "Tell me about artificial intelligence."

# Tokenize and process input
inputs = tokenizer(user_input, return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9)

# Decode and print response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("🤖 LLaMA:", response)
