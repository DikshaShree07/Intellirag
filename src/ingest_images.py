"""
Loads images from data/images, embeds them with a free pretrained
CLIP model (no training involved, just inference), and stores them
in a separate ChromaDB collection called "image_chunks".

Run this once (or whenever you add new images):
    python src/ingest_images.py
"""

import os
import glob
import chromadb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

MODEL_NAME = "openai/clip-vit-base-patch32"  # free, pretrained, ~600MB download once


def get_clip():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def embed_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    features = model.get_image_features(**inputs)
    # normalize so cosine similarity works cleanly
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.detach().numpy()[0].tolist()


def main():
    print("Loading CLIP model (first run downloads ~600MB, then it's cached)...")
    model, processor = get_clip()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection("image_chunks")

    files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + \
        glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) + \
        glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    if not files:
        print(f"No images found in {IMAGES_DIR}. Add some .jpg/.png files first.")
        return

    ids, embeddings, metadatas = [], [], []
    for path in files:
        fname = os.path.basename(path)
        print(f"  Embedding {fname}...")
        emb = embed_image(model, processor, path)
        ids.append(fname)
        embeddings.append(emb)
        metadatas.append({"source": fname, "path": path})

    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print(f"Done. Stored {len(ids)} image embeddings in ChromaDB at {DB_DIR}")


if __name__ == "__main__":
    main()
