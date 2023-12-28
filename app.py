import streamlit as st
from io import BytesIO

prg=9.090909090909091
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
    with st.form(key='my_form1'):
        st.header("General Information")
        st.subheader('1. Details of Parties:')
        st.markdown('First Party Details')
        first_name=st.text_input("Full name of the 1st party:")
        bar = st.progress(9.090909090909091)
        first_age=st.text_input("Age of the 1st party:")
        bar = st.progress(9.090909090909091*2)
        first_address=st.text_input("Residential address of the 1st party:")
        bar = st.progress(9.090909090909091*3)
        first_aadhar=st.text_input("Aadhar Number:")
        bar = st.progress(9.090909090909091*4)
        if first_aadhar!="":
            st.write(f'So the Memorandum of Understanding agreement is between {first_name},{first_age},{first_address}')
        if st.form_submit_button('Next'):
            st.markdown('Second Party Details')
            second_name=st.text_input("Full name of the 2nd party:")
            if first_name == second_name:
                st.write("Both the parties name cannot be the same")
            second_age=st.text_input("Age of the 2nd party:")
            second_address=st.text_input("Residential address of the 2nd party:")
            second_aadhar=st.text_input("Aadhar Number of the 2nd prty:")
            if second_aadhar!="":
                st.write(f'So the MOU agreement is between {first_name},{first_age},{first_address} and {second_name},{second_age},{second_address}')
                st.write('understood')
                st.subheader('2. Date and Location:')
                st.date_input('Date of Aggreement:')
                st.text_input('Location of Agreement:')
                st.checkbox('Are you sure, the entered details are correct!')
                if st.form_submit_button('Submit'):
                    st.header('Thank you for your time')
                    st.balloons()

#st.chat_input("Say something")
bar = st.progress(50)     
  
    # Dictionary to store responses
#responses = {}

#with st.sidebar:
    #st.radio('Select one:', [1, 2])
if __name__ == "__main__":
    main()
