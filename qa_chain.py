import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # For Groq LLM

groq_api_key = "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"  # Use env var in production

def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def create_qa_chain_from_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama3-70b-8192"
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
