from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from flask import Flask, request, jsonify


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "shl-assessment"


pc = PineconeClient(api_key=PINECONE_API_KEY)


existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

docsearch = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text"
)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_index():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query missing"}), 400

    results = docsearch.similarity_search(query, k=3)
    output = [{"text": r.page_content} for r in results]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
