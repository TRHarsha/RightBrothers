import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Define the function recommend_lawyers
def recommend_lawyers(user_preferences, data, specialization_matrix, languages_matrix, tfidf_specialization, tfidf_languages):
    # Initialize weights for each preference
    weights = {
        'Specialization': 0.25,
        'Languages Spoken': 0.2,
        'Availability': 0.1,
        'Gender': 0.1,
        'Location': 0.1,
        'Ratings': 0.1,
        'Experience': 0.05,
        'Age': 0.1
    }

    # Vectorize user preferences
    user_specialization = tfidf_specialization.transform([user_preferences.get('Specialization', '')])
    user_languages = tfidf_languages.transform([user_preferences.get('Languages Spoken', '')])

    # Calculate similarity scores for Specialization and Languages Spoken
    specialization_scores = np.dot(specialization_matrix, user_specialization.T).toarray().flatten()
    languages_scores = np.dot(languages_matrix, user_languages.T).toarray().flatten()

    # Combine all similarity scores with appropriate weights
    similarity_scores = (
        weights['Specialization'] * specialization_scores +
        weights['Languages Spoken'] * languages_scores
    )

    # Normalize similarity scores
    normalized_scores = similarity_scores / similarity_scores.sum()

    # Add scores for other preferences
    for key, weight in weights.items():
        if key not in ['Specialization', 'Languages Spoken']:
            if key in user_preferences and user_preferences[key] is not None:
                pref_value = user_preferences[key]
                data_value = data[key]
                if isinstance(pref_value, str):
                    similarity_scores += weight * (data_value == pref_value).astype(int)
                elif isinstance(pref_value, (int, float)):
                    similarity_scores += weight * (1 - (data_value - pref_value).abs() / (data_value.max() - data_value.min()))

    # Combine all scores
    data['Similarity'] = similarity_scores

    # Sort by similarity scores and get top 5 recommended lawyers
    recommended_lawyers = data.sort_values(by='Similarity', ascending=False).head(5)
    return recommended_lawyers

# Load the saved model components from the pickle file
pickle_model_path = 'C:\\Users\\bhuvan\\Desktop\\harsha\\AILegalAssistant-main\\AILegalAssistant-main\\Recommendation_System.pkl'
with open(pickle_model_path, 'rb') as f:
    model_components_loaded = pickle.load(f)

# Extract components from the loaded pickle file
recommend_lawyers_loaded = model_components_loaded['recommend_lawyers']
data_loaded = model_components_loaded['data']
specialization_matrix_loaded = model_components_loaded['specialization_matrix']
languages_matrix_loaded = model_components_loaded['languages_matrix']
tfidf_specialization_loaded = model_components_loaded['tfidf_specialization']
tfidf_languages_loaded = model_components_loaded['tfidf_languages']

# Streamlit app
st.title('Lawyer Recommender System')

with st.form("user_preferences_form"):
    age = st.number_input('Age', min_value=25, max_value=86)
    specialization = st.selectbox('Specialization', [
        "Corporate Law", "Criminal Law", "Banking Law", "Civil Law", "Cyber Law",
        "Labour Law", "Administrative Law", "Tax Law", "Intellectual Property Law",
        "Business Law", "Commercial Law", "Media Law", "Maritime Law",
        "Environmental Law", "Competition Law", "Air and Space Law", "Consumer Law",
        "Energy Law", "International Law", "Human Rights Law", "Real Estate Law",
        "Patent Law"
    ])
    languages = st.selectbox('Languages Spoken', [
        "English", "English, Hindi", "English, Kannada", "Hindi, Kannada", "Kannada",
        "Hindi", "Telugu", "Telugu, Kannada", "Telugu, Hindi", "Telugu, English",
        "Telugu, English, Hindi, Kannada", "Tamil, English", "Tamil, Kannada",
        "Tamil, Hindi, English", "Tamil, English, Kannada, Hindi"
    ])
    availability = st.selectbox('Availability', [
        "9:00 AM to 5:00 PM", "10:00 AM to 6:00 PM", "11:00 AM to 7:00 PM", "2:00 PM to 9:00 PM"
    ])
    gender = st.selectbox('Gender', ["Male", "Female", "Transgender"])
    location = st.selectbox('Location', [
        "Banashankari 2nd Stage, Bengaluru-560070", "Banashankari 3rd Stage, Bengaluru-560085",
        "Jayanagar 8th Block, Bengaluru-560082", "Padmanabhanagar, Bengaluru-560070",
        "Kadirenahalli, Bengaluru-560070", "Teachers Colony II Stage, Bengaluru-560078",
        "Yarab Nagar, Bengaluru-560078", "Karesandra, Bengaluru-560085", "Kaveri Nagar, Bengaluru-560085",
        "Hanumanthanagar, Bengaluru-560019", "Byatrayanapura, Bengaluru-560026", "Avalahalli, Bengaluru-560026",
        "Srinagar, Bengaluru-560050", "Nagendra Block, Bengaluru-560050", "Kalidasa Layout, Bengaluru-560050",
        "Raghavendra Block, Bengaluru-560050", "Brindavan Nagar, Bengaluru-560026", "Srinivasa Nagar, Bengaluru-560050",
        "Vidyapeetha, Bengaluru-560085", "SBM colony, Bengaluru-560085", "Ashok Nagar, Bengaluru-560050"
    ])
    ratings = st.slider('Ratings', min_value=1, max_value=5)
    experience = st.slider('Experience', min_value=0, max_value=50)
    fee_structure = st.number_input('Fee Structure (per hour)', min_value=200, max_value=5000)

    submitted = st.form_submit_button("Submit")
    if submitted:
        user_preferences = {
            'Specialization': specialization,
            'Languages Spoken': languages,
            'Availability': availability,
            'Gender': gender,
            'Location': location,
            'Ratings': ratings,
            'Experience': experience,
            'Age': age,
            'Fee Structure': fee_structure
        }

        recommended_lawyers = recommend_lawyers(user_preferences, data_loaded, specialization_matrix_loaded, languages_matrix_loaded, tfidf_specialization_loaded, tfidf_languages_loaded)
        recommended_lawyers_list = recommended_lawyers.to_dict(orient='records')

        st.header('Recommended Lawyers')


        # Function to create a lawyer card with a mailto link and custom text colors
        def create_lawyer_card(name, specialization, languages, location, gender, availability, age, fee,
                               email_address):
            card_html = f"""
            <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;'>
                <h3 style='color: #333;'>{name}</h3>
                <p><b style='color: #4CAF50;'>Specialization:</b> <span style='color: #666;'>{specialization}</span></p>
                <p><b style='color: #4CAF50;'>Languages Spoken:</b> <span style='color: #666;'>{languages}</span></p>
                <p><b style='color: #4CAF50;'>Location:</b> <span style='color: #666;'>{location}</span></p>
                <p><b style='color: #4CAF50;'>Gender:</b> <span style='color: #666;'>{gender}</span></p>
                <p><b style='color: #4CAF50;'>Availability:</b> <span style='color: #666;'>{availability}</span></p>
                <p><b style='color: #4CAF50;'>Age:</b> <span style='color: #666;'>{age}</span></p>
                <p><b style='color: #4CAF50;'>Fee Structure:</b> <span style='color: #666;'>Rs. {fee}/hour</span></p>
                <a href="mailto:{email_address}" style='background-color: #4CAF50; border: none; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; border-radius: 5px;'>Contact Us</a>
            </div>
            """
            return card_html


        # Display recommended lawyers in card view
        for i in range(0, len(recommended_lawyers_list), 2):
            cols = st.columns(2)
            for col, lawyer in zip(cols, recommended_lawyers_list[i:i + 2]):
                col.markdown(create_lawyer_card(
                    lawyer['Name'],
                    lawyer['Specialization'],
                    lawyer['Languages Spoken'],
                    lawyer['Location'],
                    lawyer['Gender'],
                    lawyer['Availability'],
                    lawyer['Age'],
                    lawyer['Fee Structure'],
                    "writeforus@legaease.com"  # Email address for contact
                ), unsafe_allow_html=True)