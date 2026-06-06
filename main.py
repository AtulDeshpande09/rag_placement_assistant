# all the imports
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch

class LocalEmbeddings:
    def __init__(self, model_path="./models/embeddings"):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embeddings = LocalEmbeddings("./models/embeddings")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Loading model - MISTRAL 7B INSTRUCT
model_name = "./models/mistral_7b_instruct" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure pad token is set (Mistral sometimes lacks it, which causes warnings/errors in batch generation)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

# load vector DB
CHROMA_PATH = "chroma"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# --- OPTIMIZED PROMPT TEMPLATE ---
# We use a System message to set strict behavioral rules for Mistral
SYSTEM_PROMPT = """You are an expert technical interviewer and AI assistant. 
Your task is to generate highly relevant, company-specific, and role-specific interview questions based strictly on the provided context."""

PROMPT_TEMPLATE = """
Use ONLY the following context to generate your answers. Do not use outside knowledge.

<context>
{context}
</context>

---

**Task:**  
Generate exactly **20 interview questions** relevant to the role and company described in the user query below.

**Output Format:**
You must strictly follow this format for every single question:
Q1: [Question text]
A1: [Very short answer based on context, or "_Not available in context_"]
Q2: [Question text]
A2: [Very short answer based on context, or "_Not available in context_"]
...and so on up to Q20.

**CRITICAL RULES:**
1. DO NOT output any introductory text (e.g., "Here are the questions:"). Start immediately with "Q1:".
2. DO NOT output any concluding text or commentary at the end.
3. DO NOT use markdown formatting (no **bold**, no # headers, no bullet points). Use plain text only.
4. If the answer is not found in the context, the answer MUST be exactly: _Not available in context_
5. Keep questions clear, realistic, and professional.

---

User Query: {query}

Now produce the final output:
"""

def generate_interview_response(query_text):
    # 1. Retrieve context
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # If context is empty, handle it gracefully
    if not context_text.strip():
        return "No relevant context found in the database for this query."

    user_prompt = PROMPT_TEMPLATE.format(context=context_text, query=query_text)

    # 2. Format the prompt using Mistral's specific chat template (System + User)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Tokenize inputs for direct torch generation
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # 4. Generate answer using direct torch generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,       # Sufficient for 20 Q&As
            do_sample=True,
            temperature=0.15,          # Lowered for strict formatting and factual accuracy
            top_p=0.85,
            repetition_penalty=1.15,   # Increased to prevent looping
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # Optional: Uncomment to see generation token-by-token in Colab console
            # streamer=TextStreamer(tokenizer, skip_prompt=True), 
        )

    # 5. Decode ONLY the newly generated tokens (skip the prompt)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Basic cleanup in case the model still adds a tiny bit of conversational filler
    if response.lower().startswith("here are"):
        response = response.split(":", 1)[-1].strip()
        
    return response
