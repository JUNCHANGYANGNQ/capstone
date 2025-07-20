from dotenv import load_dotenv
load_dotenv()
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


app = Flask(__name__, static_folder="frontend/dist copy", static_url_path="/")
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

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(os.path.join(base_path, "vector_store"))

    return jsonify({"message": "Embedding successful", "university": university_id}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    university_id = data.get("university")
    question = data.get("question")
    if not university_id or not question:
        return jsonify({"error": "Missing university or question"}), 400

    try:
        db_path = f"data/{university_id}/vector_store"
        embedding = OpenAIEmbeddings()
        vector_db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=retriever
        )
        answer = qa_chain.run(question)
        return jsonify({"answer": answer}), 200

    except Exception as e:
        print("Error during QA:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)

