import os
os.environ["HF_HOME"] = "D:\\hf_models"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_models\\transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\hf_models\\hub"
from dotenv import load_dotenv
load_dotenv()
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from huggingface_hub import snapshot_download
from langchain.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


app = Flask(__name__, static_folder="frontend", static_url_path="/")
CORS(app)

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/embed", methods=["POST"])
def embed_documents():
    university_id = request.form.get("university")
    file = request.files.get("file")
    if not university_id or not file:
        return jsonify({"error": "Missing university ID or file"}), 400

    base_path = f"data/{university_id}"
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, file.filename)
    file.save(file_path)

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(os.path.join(base_path, "vector_store"))

    return jsonify({"message": "Embedding successful", "university": university_id}), 200



MODEL_MAP = {
    "mixtral": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma": "google/gemma-1.1-9b-it",
    "qwen":"Qwen/Qwen1.5-1.8B-Chat"
}


def load_llm(model_key):
    login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    model_id = MODEL_MAP.get(model_key, MODEL_MAP["qwen"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",            
        torch_dtype="auto",           
        trust_remote_code=True        
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,  
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_chat_prompt(context: str, question: str) -> str:
    return f"""<|im_start|>system
You are an assistant for a university QA system. Use the context to answer the question. Be concise and do not make up information.<|im_end|>
<|im_start|>user
<context>
{context}
</context>
Question: {question}<|im_end|>
<|im_start|>assistant
"""


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    university_id = data.get("university")
    question = data.get("question")
    model_key = data.get("model", "qwen")

    if not university_id or not question:
        return jsonify({"error": "Missing university or question"}), 400

    try:
        
        db_path = f"data/{university_id}/vector_store"
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        
        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in docs])

        
        prompt = build_chat_prompt(context_text, question)

        
        llm_pipeline = load_llm(model_key).pipeline

        
        raw_output = llm_pipeline(prompt)[0]["generated_text"]

        
        if "<|im_start|>assistant" in raw_output:
            answer = raw_output.split("<|im_start|>assistant")[-1].strip()
        else:
            answer = raw_output.strip()

        return jsonify({"answer": answer}), 200

    except Exception as e:
        print("Error during QA:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
