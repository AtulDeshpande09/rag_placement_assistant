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
You are an AI assistant that generates **interview preparation material**.

Use ONLY the following retrieved context when generating your answer:

{context}

---

**Task:**  
Based on the context, generate a list of **10 to 20 interview questions** that are **relevant to the role and company** described in the user query below.

- Each question should be **concise and realistic**.
- If answers are clearly found in the context, provide **short precise answers (1-3 sentences max)** below each question.
- If an answer is **not clearly present** in the context, **do NOT invent or assume**. Instead, write:  
  `_No reliable answer in context_`

---

**User Query:**  
{query}

Now produce the final output in the following structured format:

Q1: <question>  
A1: <answer or _No reliable answer in context_>

Q2: <question>  
A2: <answer>

...
"""


def generate_interview_response(query_text):
    # Retrieve context
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])

    prompt = PROMPT_TEMPLATE.format(context=context_text, query=query_text)

    # Generate answer using your text-generation pipeline
    response = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.3)[0]["generated_text"]

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]

    final_response = f"{response}\n\nSources: {sources}"
    return final_response
