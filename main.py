# all the imports
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline , BitsAndBytesConfig
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
    load_in_4bit=True,                # 4-bit quantization
    bnb_4bit_use_double_quant=True,   # double quantization for extra compression
    bnb_4bit_quant_type="nf4",        # better precision scheme
    bnb_4bit_compute_dtype=torch.float16
)

# Loading model
model_name = "./models/phi_mini" 

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                # automatically assign GPU/CPU
    quantization_config=bnb_config,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# load vector DB

CHROMA_PATH = "chroma"

db = Chroma(persist_directory = CHROMA_PATH,
            embedding_function= embeddings)


PROMPT_TEMPLATE = """
You are an AI assistant that generates company & role specific interview preparation questions.

Use ONLY the following context when generating your answers:

{context}

---

**Task:**  
Generate **20 interview questions** relevant to the role and company described in the user query below.

- Write the output in the format:
Q1: <question>
A1: <very short answer or "_Not available in context_">

- Keep questions clear and realistic.
- If the answer is not found in the context, write: _Not available in context_
- DO NOT stop mid-response.
- DO NOT add extra commentary.

---

User Query: {query}

Now produce the final output:
"""



def generate_interview_response(query_text):
    # Retrieve context
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])

    prompt = PROMPT_TEMPLATE.format(context=context_text, query=query_text)

    # Generate answer using your text-generation pipeline
    response = pipe(
        prompt,
        max_new_tokens=600,       # increased
        do_sample=True,
        temperature=0.3,          # keeps answers focused
        top_p=0.9,
        repetition_penalty=1.1
        )[0]["generated_text"]

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]

    final_response = f"{response}"
    return final_response
