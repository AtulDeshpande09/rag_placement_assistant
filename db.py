from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


DATAPATH = "data/"
DBPATH = "chroma"


### Load documents 
def load_documents(DATAPATH=DATAPATH):
    loader = DirectoryLoader(DATAPATH,glob="*.md")
    document = loader.load()
    return document


### divide them into chunks

def split_text(documents):

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 500,
            length_function = len,
            add_start_index = True
            )
    chunks = text_splitter.split_documents(documents)

    return chunks

if __name__ == "__main__":

    documents = load_documents()
    print(f"Loaded Documents from {DATAPATH}")
    
    chunks = split_text(documents)
    print("Docs divided into chunks")
    

    embedding_function = SentenceTransformer("./models/embeddings")

    db = Chroma.from_documents(
            chunks,
            embedding_function,
            persist_directory = DBPATH
            )
    db.persist()
    print(f"Vector DB created and saved at {DBPATH}")
