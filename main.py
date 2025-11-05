# all the imports
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# Loading model
model_name = "./models/phi_mini" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda" )

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# load vector DB

CHROMA_PATH = "chroma"

"""
embedding_function = HuggingFaceEmbeddings(
    model_name="./models/embeddings"
)
"""

embedding_function = SentenceTransformer("./models/embeddings")

db = Chroma(persist_directory = CHROMA_PATH,
            embedding_function= embedding_function)


def generate_interview_response(query_text):
    # Retrieve context
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---
    Now answer the user query:

    {query}
    """

    prompt = PROMPT_TEMPLATE.format(context=context_text, query=query_text)

    # Generate answer using your text-generation pipeline
    response = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.3)[0]["generated_text"]

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]

    final_response = f"{response}\n\nSources: {sources}"
    return final_response
