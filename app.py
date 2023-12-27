import streamlit as st
from io import BytesIO

# Function to generate PDF based on questionnaire responses
def generate_pdf(responses):
    buffer = BytesIO()

    # Create a PDF document
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 800, "Generated PDF based on Questionnaire")

    # Insert responses into the PDF
    y_position = 780
    for question, answer in responses.items():
        y_position -= 20
        p.drawString(100, y_position, f"{question}: {answer}")

    # Save the PDF
    p.showPage()
    p.save()

    buffer.seek(0)
    return buffer

# Streamlit app
def main():
    st.title("Right Brothers")
    st.subheader('Legal Brothers for Life', divider='rainbow')
    st.header("Questionnaire and PDF Generator")


    #Outline 
    st.header("General Information")
    st.subheader('1. Details of Parties:')
    questions = [
        "Full name of the 1st party:",
        "Age of the 1st party:",
        "Residential address of the 1st party:",
        "Aadhar Number:"
    ]

    # Dictionary to store responses
    responses = {}


if __name__ == "__main__":
    main()
