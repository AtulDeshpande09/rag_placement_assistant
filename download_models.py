# Run this once in a Colab cell if bitsandbytes isn't already installed:
# !pip install -q bitsandbytes accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import os
import torch

# Mistral 7B Instruct v0.3 is highly optimized and fits well on Colab's free GPU
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

save_dir = "./models/mistral_7b_instruct"
embd_dir = "./models/embeddings"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(embd_dir, exist_ok=True)
print("Folders created!!!\n")

# 4-bit quantization config to keep VRAM usage ~4-5GB (fits easily on T4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=False,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically places layers on GPU
)
print("Downloaded & loaded model with 4-bit quantization!!!\n")

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Saved model at {save_dir}\n")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder.save(embd_dir)

print(f"Saved embedding model at {embd_dir}")
