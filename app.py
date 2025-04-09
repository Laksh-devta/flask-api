import os
import json
import time
from pathlib import Path
from collections import OrderedDict
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


PRODUCTS_JSON_PATH = Path("JSONs/products.json")

# Pinecone
INDEX_NAME = "shl-product-index"
DIMENSION = 768
REGION = "us-east-1"


SIMILARITY_THRESHOLD = 0.5
MAX_RECOMMENDATIONS = 10


app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False


def load_products(filepath: Path):
    """
    Loads the product JSON data and returns a dictionary mapping product ID to product details.
    """
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


products_db = load_products(PRODUCTS_JSON_PATH)


pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index '{INDEX_NAME}' ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

    while INDEX_NAME not in pc.list_indexes().names():
        time.sleep(1)
else:
    print(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():

    try:
        data_in = request.get_json(force=True)
        query = data_in.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing or empty 'query' field."}), 400

        query_embedding = embedder.embed_query(query)

        search_response = index.query(
            vector=query_embedding,
            top_k=MAX_RECOMMENDATIONS,
            include_metadata=False
        )

        # Filter matches based on the similarity threshold.
        filtered_matches = [
            match for match in search_response.get("matches", [])
            if match.get("score", 0) >= SIMILARITY_THRESHOLD
        ]

        recommended = []
        for match in filtered_matches:
            product_id = match["id"]
            product = products_db.get(product_id)
            if product:
                rec = OrderedDict([
                    ("url", product.get("url", "")),
                    ("adaptive_support", product.get("adaptive_support", "")),
                    ("description", product.get("description", "")),
                    ("duration", int(product.get("duration", 0))),
                    ("remote_support", product.get("remote_support", "")),
                    ("test_type", product.get("test_type", []))
                ])
                recommended.append(rec)

        if not recommended:
            return jsonify({"error": "No recommendations found above the similarity threshold."}), 404

        # json order
        response_json = json.dumps({"recommended_assessments": recommended},
                                   ensure_ascii=False,
                                   indent=2,
                                   sort_keys=False)
        return app.response_class(response=response_json, status=200, mimetype="application/json")

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
