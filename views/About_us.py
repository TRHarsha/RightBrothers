import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os

# Function to load an image from a URL
def load_image(path):
    try:
        # Check if it's a URL (contains 'http')
        if path.startswith('http'):
            response = requests.get(path)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        # If it's a local path
        elif os.path.exists(path):
            return Image.open(path)
        else:
            raise ValueError("File path does not exist.")
    except Exception as e:
        st.error(f"Error loading image: {e}")
    return None

# Example image URLs
images = {
    "main": "https://st2.depositphotos.com/3837271/9109/i/450/depositphotos_91090624-stock-photo-wooden-blocks-with-the-text.jpg", 
    "main1": "https://img.freepik.com/free-photo/still-life-with-scales-justice_23-2149776024.jpg",
    "team": [
        {
        "img": "C:/Users/bhuvan/Desktop/harsha/AILegalAssistant-main/AILegalAssistant-main/img/eeshanvc.jpg",
        "name": "Eeshan V C",
        "usn": "4PM21AI014",
        "email": "eeshanvc@gmail.com",
    },
    {
        "img": "img/harshatr.png",
        "name": "Harsha T R",
        "usn": "4PM21AI017",
        "email": "harshatr.work@gmail.com",
    },
    {
        "img": "img/pass.png",
        "name": "Prabhanjana K",
        "usn": "4PM21AI026",
        "email": "prabh.bhat12@gmail.com",
    },
    {
        "img": "img/prathibha.jpg",
        "name": "Prathibha P",
        "usn": "4PM21AI028",
        "email": "prathibhasagar@gmail.com",
    }
    ],
}

# Load main images
main_image = load_image(images["main"])
main_image1 = load_image(images["main1"])

st.header("⚖️ Introducing Right Brothers")
st.write("---")
st.header("😃 About Us!")
st.write("##")

# About Us Section
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Revolutionizing Legal Consultation with AI.")
        st.write(
            """
            At Right Brothers, we are dedicated to transforming the legal industry by leveraging the power of AI 
            to make legal consultation more accessible, transparent, and efficient. Our mission is to simplify 
            legal processes, assisting individuals and businesses in navigating complex legal landscapes with ease. 
            """
        )
    with col2:
        if main_image1:
            st.image(main_image1, use_column_width=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if main_image:
            st.image(main_image, caption="AI-Driven Legal Solutions", use_column_width=True)
    with col2:
        st.header("AI-Powered Legal Assistance at Your Fingertips.")
        st.write(
            """
            Right Brothers' platform utilizes cutting-edge AI to automate and streamline the process of legal consultation. 
            Whether it's creating legal notices, researching case laws, or assisting in legal document preparation, 
            our AI system ensures accuracy, speed, and ease of use, empowering users to handle their legal affairs seamlessly.
            """
        )

st.write("---")
st.header("Meet Our Team!")
st.write("##")

# Team Section
for member in images["team"]:
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            profile_image = load_image(member["img"])
            if profile_image:
                st.image(profile_image, use_column_width=True)
        with col2:
            st.write(f"**Name:** {member['name']}")
            st.write(f"**USN:** {member['usn']}")
            st.write(f"**Email:** {member['email']}")

# Footer
st.markdown(
    """
    <style>
        .footer {
            position: relative;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            padding: 10px 0;
            text-align: center;
            font-size: 12px;
            color: #f0f0f0;
        }
    </style>
    <div class="footer">
        <p>© 2024-2025 RightBrothers, Inc. · <a href="#">Privacy</a> · <a href="#">Terms</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
