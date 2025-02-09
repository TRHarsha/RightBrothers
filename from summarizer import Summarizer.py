from summarizer import Summarizer
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract

# Load models
@st.cache_resource
def load_models():
    return Summarizer(), SentenceTransformer('all-MiniLM-L6-v2')

bert_summarizer, model = load_models()

uploaded_file = st.file_uploader("Upload a case file (PDF, DOCX, or Image):", type=["pdf", "docx", "jpg", "jpeg", "png"])

def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = "\n\n".join([page.extract_text().strip() for page in pdf_reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(file)
        text = "\n\n".join([paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text])
    elif file.type.startswith("image/"):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    else:
        st.error("Unsupported file type.")
    return text

# Load dataset and embeddings (Dummy Example)
# Load CSV and JSON, then create embeddings for the entire dataset
@st.cache_data
def load_data_and_embeddings():
    # Load CSV file
    csv_data = pd.read_csv("C:/Users/bhuvan/Desktop/harsha/AILegalAssistant-main/AILegalAssistant-main/leg_dataset.csv")
    
    # Load JSON file
    json_data = pd.read_json("C:/Users/bhuvan/Desktop/harsha/AILegalAssistant-main/AILegalAssistant-main/courtcase.json")
    
    # Combine CSV and JSON data
    combined_data = pd.concat([csv_data, json_data], ignore_index=True)
    
    # Create a single content column by joining all columns
    combined_data['content'] = combined_data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    
    # Generate embeddings
    embeddings = model.encode(combined_data['content'].tolist(), show_progress_bar=True)
    return combined_data, embeddings

data, embeddings = load_data_and_embeddings()

# Efficient Semantic Search using Embeddings
def search_with_embeddings(query, embeddings, data, top_k=5):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = scores.argsort()[-top_k:][::-1]  # Get indices of top k results
    return data.iloc[top_k_indices], scores[top_k_indices]

# Generate a summary using BERT Extractive Summarization
def generate_summary(text, max_sentences=3):
    return bert_summarizer(text, num_sentences=max_sentences)

# Inside the Streamlit UI section:
if uploaded_file:
    extracted_text = extract_text_from_file(uploaded_file)
    
    if extracted_text:
        st.write("**Extracted Text from Uploaded File:**")
        st.write(extracted_text)
        
        # Perform semantic search
        results, scores = search_with_embeddings(extracted_text, embeddings, data, top_k=5)
        
        # Display search results
        st.write("### Search Results:")
        for i, (index, row) in enumerate(results.iterrows()):
            case_text = row['content']
            summary = generate_summary(case_text, max_sentences=2)
            st.write(f"**Result {i+1} (Similarity: {scores[i]:.2f}):**")
            st.write(summary)
            with st.expander("View Full Case Text"):
                st.write(case_text)
