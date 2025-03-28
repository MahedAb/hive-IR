import os
import json
import faiss
import torch
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import numpy as np



# Step 1: Load MedQuAD XML data
chunks = []
data_root = Path("MedQuAD")  # Adjust if needed

for subfolder in data_root.glob("*_QA"):
    for file in subfolder.glob("*.xml"):
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            q_el = root.find(".//Question")
            a_el = root.find(".//Answer")

            if q_el is not None and a_el is not None:
                question = q_el.text.strip() if q_el.text else ""
                answer = a_el.text.strip() if a_el.text else ""

                if question and answer:
                    chunks.append({
                        "text": f"Q: {question}\nA: {answer}",
                        "meta": {
                            "source": subfolder.name,
                            "file": file.name
                        }
                    })
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Step 2: Load embedding models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 3: Encode MedQuAD text with CLIP
clip_text_embeddings = []
for chunk in chunks:
    inputs = clip_processor(text=[chunk["text"]], return_tensors="pt", padding=True, truncation=True)
    embedding = clip_model.get_text_features(**inputs).detach().cpu().numpy()[0]
    clip_text_embeddings.append(embedding)


clip_text_embeddings = np.array(clip_text_embeddings)
text_index = faiss.IndexFlatL2(clip_text_embeddings.shape[1])
text_index.add(clip_text_embeddings)

# Step 4: Load DermNet dataset and embed sample images
image_chunks = []
image_embeddings = []
dermnet = load_dataset("dermnet", split="train")

for i in range(100):  # Limit for speed/test
    item = dermnet[i]
    image = item["image"]
    label = item["label"]
    inputs = clip_processor(images=image, return_tensors="pt")
    embedding = clip_model.get_image_features(**inputs).detach().cpu().numpy()[0]
    image_chunks.append({"image": image, "label": label})
    image_embeddings.append(embedding)

image_embeddings = np.array(image_embeddings)
image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
image_index.add(image_embeddings)

# Step 5: Load generation model
generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=0 if torch.cuda.is_available() else -1)

# Step 6: Choose query (text or image)
is_image_query = False  # Set True to test image

if is_image_query:
    url = "https://huggingface.co/datasets/dermnet/resolve/main/images/acne/dermnet_image_1.jpg"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    image_embedding = clip_model.get_image_features(**inputs).detach().cpu().numpy()
    D, indices = text_index.search(image_embedding, k=3)
    query = "What condition does this image suggest?"
else:
    query = "What are the symptoms and treatment of Kawasaki disease?"
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    query_embedding = clip_model.get_text_features(**inputs).detach().cpu().numpy()
    D, indices = text_index.search(query_embedding, k=3)

# Step 7: Build context
context = "\n".join([f"({chunks[i]['meta']['source']}) {chunks[i]['text']}" for i in indices[0]])
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

# Step 8: Trim long input
max_prompt_tokens = 1500
if len(prompt.split()) > max_prompt_tokens:
    prompt = " ".join(prompt.split()[:max_prompt_tokens])

# Step 9: Generate answer
output = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]['generated_text']

# Step 10: Display
print("----- Context Used -----\n")
print(context)
print("\n----- Generated Answer -----\n")
print(output[len(prompt):])
