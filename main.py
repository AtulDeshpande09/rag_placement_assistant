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

if __name__ == '__main__':


    query_text = input("Enter You Query : ")

    results = db.similarity_search_with_score(query_text, k=3)


    PROMPT_TEMPLATE = """
    Answer the question based only on the following context : {context}
    
    ---
    Answer the question based on the above context : {query}
    """

    # Actual Prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc,_score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, query = query_text)
    # model 

    model = "some model"

    response_text = model.predict(prompt)

    sources = [doc.metadata.get("Source",None) for doc, _score in results]
    
    final_response = f"Response : {response_text} \nSource : {sources}"
    print("Done!!!")
