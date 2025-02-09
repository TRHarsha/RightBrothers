import streamlit as st
from PIL import Image
# Set page config
st.set_page_config(page_title="Write for Us - RightBrothers", layout="wide")
st.write("---")
st.title("Contact Us Form")
st.write("##")

# Email input
email = st.text_input("Email")

# Phone number input
phone_number = st.text_input("Phone Number")

# Checkboxes for different categories
st.subheader("Please select applicable categories:")
is_individual = st.checkbox("Are you an Individual?")
is_small_business_owner = st.checkbox("Are you a Small Business Owner?")
is_startup_owner = st.checkbox("Are you a Startup Owner?")
is_legal_professional = st.checkbox("Are you a Legal Professional?")
is_student_or_researcher = st.checkbox("Are you a Student or Researcher?")
is_ngo_or_government_agency = st.checkbox("Are you an NGO or Government Agency?")

# Text area for elaborating concerns
concerns = st.text_area("Elaborate your concern")

# Submit button
submitted = st.button("Submit")

# Handling submission
if submitted:
    # You can add your own custom logic here, such as sending an email or saving the form data to a database
    st.write("Form submitted successfully!")


# Footer
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #000000;
            padding: 5px 0;
            text-align: center;
            font-size: 12px;
            color: #f0f0f0;
        }
    </style>
    <div class="footer">
        <p>© 2023-2024 LegaEase, Inc. · <a href="#">Privacy</a> · <a href="#">Terms</a></p>
    </div>
    """,
    unsafe_allow_html=True
)