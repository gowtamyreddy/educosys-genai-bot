import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, Docx2txtLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Groq API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "llama3-70b-8192"

# Your real note paths
BASE_PATHS = {
    "typed": "Dataset",
    "handwritten": "Dataset\Handwritten_notes",
}

# Load documents from chosen type(s)
def load_documents_by_type(selected_types):
    all_docs = []
    for note_type in selected_types:
        doc_folder = BASE_PATHS[note_type]
        for filename in os.listdir(doc_folder):
            path = os.path.join(doc_folder, filename)
            if filename.endswith(".txt"):
                loader = TextLoader(path)
            elif filename.endswith(".pdf"):
                loader = PyMuPDFLoader(path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            else:
                continue
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# Load or create vector store
@st.cache_resource
def load_vectorstore(selected_types):
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    store_name = "_".join(sorted(selected_types))
    vector_path = f"faiss_index_{store_name}"

    if not os.path.exists(vector_path):
        st.info(f"📚 Creating FAISS vectorstore from: {', '.join(selected_types)}")
        docs = load_documents_by_type(selected_types)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, embedding)
        db.save_local(vector_path)
    else:
        db = FAISS.load_local(vector_path, embedding, allow_dangerous_deserialization=True)
    return db

# Answer only from context
def ask_with_context(context, question):
    prompt = f"""You are a helpful tutor. Use ONLY the context below to answer the question.
If the answer is not present, say: "I don't know."

Context:
{context}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Educosys AI Q&A", layout="centered")
st.title("📘 Educosys Notes Chatbot (Groq + LLaMA3)")
note_type = st.radio("📂 Choose source:", ["typed", "handwritten", "both"], index=2, horizontal=True)

type_map = {
    "typed": ["typed"],
    "handwritten": ["handwritten"],
    "both": ["typed", "handwritten"]
}

query = st.text_input("🔍 Ask a question from your notes:")

if query:
    with st.spinner("Thinking..."):
        db = load_vectorstore(type_map[note_type])
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        st.markdown("### 🔎 Source Chunks:")
        for i, doc in enumerate(docs[:3]):
            st.text_area(f"Chunk {i+1}", doc.page_content[:400], height=150)

        answer = ask_with_context(context, query)

    st.success("✅ Answer:")
    st.write(answer)
