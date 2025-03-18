import base64
import os
import uvicorn
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost:5173",
    "https://localhost:3000",
    "http://localhost:3000",
    "http://localhost",
    "https://localhost",
    "http://localhost:8000",
    "https://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Read and process uploaded text files
def read_uploaded_text(file):
    content = file.file.read().decode("utf-8")
    return [line.strip() for line in content.split("\n") if line.strip()]

@app.get("/")
async def read_root():
    return {"message": "Welcome to the OBE Validator!"}

@app.post("/validate_obe/")
async def validate_obe(
    syllabus: UploadFile = File(...),
    questions: UploadFile = File(...),
    threshold: float = Form(...)
):
    try:
        # Read syllabus and questions
        syllabus_chunks = read_uploaded_text(syllabus)
        questions_list = read_uploaded_text(questions)

        # Generate embeddings
        syllabus_embeddings = [get_bert_embedding(chunk) for chunk in syllabus_chunks]
        question_embeddings = [get_bert_embedding(question) for question in questions_list]

        # Process similarity
        results = []
        aligned_pred = []

        for question, q_emb in zip(questions_list, question_embeddings):
            similarities = [cosine_similarity(q_emb, s_emb)[0][0] for s_emb in syllabus_embeddings]
            best_match_idx = similarities.index(max(similarities))
            best_similarity = max(similarities)

            coherent = best_similarity >= threshold
            aligned_pred.append(1 if coherent else 0)

            results.append({
                "question": question,
                "best_matching_syllabus": syllabus_chunks[best_match_idx],
                "similarity_score": round(float(best_similarity), 2),
                "coherent": "Yes" if coherent else "No",
                "aligned_pred": 1 if coherent else 0
            })

        # Load actual aligned labels from static directory
        df = pd.read_excel("static/actual_aligned.xlsx")
        aligned_list = df["Aligned"].astype(int).tolist()

        accuracy = float(accuracy_score(aligned_list, aligned_pred))  # Convert to float
        precision = float(precision_score(aligned_list, aligned_pred, average="weighted"))
        recall = float(recall_score(aligned_list, aligned_pred, average="weighted"))
        f1 = float(f1_score(aligned_list, aligned_pred, average="weighted"))
        conf_matrix = confusion_matrix(aligned_list, aligned_pred)  # Don't use .tolist() yet

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Convert plot to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        buffer.close()
        plt.close()

        return {
            "results": results,
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_matrix_image": base64_img
        }

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    port = os.environ.get("PORT", "8000")  # Fetch PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=int(port))  # Ensure PORT is an int