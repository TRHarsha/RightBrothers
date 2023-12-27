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
    with st.form(key='my_form1'):
        st.markdown('First Party Details')
        first_name=st.text_input("Full name of the 1st party:")
        first_age=st.text_input("Age of the 1st party:")
        first_address=st.text_input("Residential address of the 1st party:")
        first_aadhar=st.text_input("Aadhar Number:")
        if first_aadhar!="":
            st.write(f'So the lease agreement is between {first_name},{first_age},{first_address}')
        st.form_submit_button('Next')
    with st.form(key='my_form2'):
        st.markdown('Second Party Details')
        second_name=st.text_input("Full name of the 2nd party:")
        second_age=st.text_input("Age of the 2nd party:")
        second_address=st.text_input("Residential address of the 2nd party:")
        second_aadhar=st.text_input("Aadhar Number:")
        if second_aadhar!="":
            st.write(f'So the lease agreement is between {first_name},{first_age},{first_address} and {second_name},{second_age},{second_address}')
        st.form_submit_button('Next')
  
    # Dictionary to store responses
    responses = {}

with st.sidebar:
    st.radio('Select one:', [1, 2])
if __name__ == "__main__":
    main()
