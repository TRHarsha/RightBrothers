import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
from keybert import KeyBERT
import matplotlib.pyplot as plt
import networkx as nx
from gtts import gTTS
import os
from transformers import pipeline
from summarizer import Summarizer

# Function to extract text from an uploaded file
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

# Load the model for semantic search (Sentence-BERT for efficient embeddings)
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Compact, fast model for embeddings

model = load_model()

# Load KeyBERT model for keyword extraction
@st.cache_resource
def load_keybert():
    return KeyBERT()

keybert_model = load_keybert()

# Load bert_summarizer models
@st.cache_resource
def load_models():
    return Summarizer(), SentenceTransformer('all-MiniLM-L6-v2')

bert_summarizer, model = load_models()

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
    summary_text =  bert_summarizer(text, num_sentences=max_sentences)
    return summary_text

# Extract keywords using KeyBERT
def extract_keywords(text, top_n=5):
    keywords = keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]

# Streamlit Interface
st.title("AI-Powered Legal Search Engine")

# File upload
uploaded_file = st.file_uploader("Upload a case file (PDF, DOCX, or Image):", type=["pdf", "docx", "jpg", "jpeg", "png"])

if uploaded_file:
    # Extract text from the uploaded file
    extracted_text = extract_text_from_file(uploaded_file)
    if extracted_text:
        st.write("**Extracted Text from Uploaded File:**")
        st.write(extracted_text)

        # Perform semantic search based on the extracted text
        st.write("### Search Results")
        top_k = st.slider("Number of results:", 1, 20, 5)  # Allow user to set the number of results
        results, scores = search_with_embeddings(extracted_text, embeddings, data, top_k=top_k)
        
        # Create a results table
        table_data = []
        keywords_list = []
        for i, (index, row) in enumerate(results.iterrows()):
            case_text = row['content']
            summary = generate_summary(case_text)  # Very short summary
            keywords = extract_keywords(case_text)  # Extract 5 main keywords
            keywords_list.append(keywords)
            table_data.append({
                "sl.No": i + 1,
                "Similarity Rate": f"{scores[i]:.2f}",
                "Summary": summary,
                "Keywords": ", ".join(keywords)  # Join keywords into a single string
            })
        
        # Display results in a table
        df_results = pd.DataFrame(table_data)
        st.write(df_results)
        
        # Allow user to select a result by sl.No
        selected_sl_no = st.number_input("Select a result by sl.No:", min_value=1, max_value=top_k, step=1)
        selected_index = selected_sl_no - 1  # Convert sl.No to index

        # Show detailed analysis for the selected result
        if selected_sl_no and 0 <= selected_index < len(table_data):
            selected_case = results.iloc[selected_index]['content']
            st.write("### Detailed Analysis")
            st.write("**Summary:**")
            st.write(generate_summary(selected_case))  # Slightly longer summary
            story_text=generate_summary(selected_case)
            # Extract and display keywords
            keywords = extract_keywords(selected_case)
            st.write("**Main Keywords:**")
            st.write(", ".join(keywords))
            
            # Show full text (optional)
            with st.expander("View Full Case Text"):
                st.write(selected_case)
    else:
        st.error("No text could be extracted from the uploaded file.")
else:
    st.info("Please upload a case file to begin the search.")
