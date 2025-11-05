from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

model_name = "microsoft/Phi-3-mini-4k-instruct"

save_dir = "./models/phi_mini"
embd_dir = "./models/embeddings"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(embd_dir, exist_ok=True)
print(f"Folders created!!!\n")

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False)
print(f"Downloaded model!!!\n")


tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Saved model at {save_dir}\n")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder.save(embd_dir)

print(f"Saved model at {embd_dir}")
