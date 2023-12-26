import base64
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from collections.abc import Iterable


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

    # Split the screen into two columns (full width)
    tab1, tab2 = st.tabs("Tab1", "Tab2")

    # Collect responses and update PDF on the fly
    for question in questions:
        response = tab1.text_input(question)
        responses[question] = response

        # Generate PDF on the fly
        pdf_buffer = generate_pdf(responses)

        # Display PDF in the web app
        tab2.subheader("Generated PDF:")
        media_type = "application/pdf"
        media_str = base64.b64encode(pdf_buffer.read()).decode("utf-8")
        pdf_display = f'<embed src="data:{media_type};base64,{media_str}" type="{media_type}" width="100%" height="600">'
        tab2.write(pdf_display, unsafe_allow_html=True)
        
        # Download PDF
        tab2.subheader("Download Generated PDF:")
        tab2.download_button(label="Click to Download", data=pdf_buffer, file_name="generated_pdf.pdf", key="download_pdf")

if __name__ == "__main__":
    main()
