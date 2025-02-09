import streamlit as st

# Define the main function
def main():
    st.set_page_config(page_title="Right Brothers", layout="wide")

    # --- SHARED ON ALL PAGES ---
    st.sidebar.caption("Made with ❤️ by Team Right Brothers")

    # --- PAGE SETUP ---
    about_page = st.Page(
        "views/About_us.py",
        title="About",
        icon="⚖️",
        default=True,
    )
    project_1_page = st.Page(
        "views/judgmentPred.py",
        title="Judgment Predictor",
        icon="🧑‍⚖️",
    )
    project_2_page = st.Page(
        "views/docGen.py",
        title="Legal Doc Generator",
        icon="📝",
    )
    project_3_page = st.Page(
        "views/chatbotLegalv2.py",
        title="Chat Bot",
        icon="💬",
    )
    project_4_page = st.Page(
        "views/Recommendation_System.py",
        title="Recommendation System",
        icon="🕵️",
    )
    project_5_page = st.Page(
        "views/SearchEngine.py",
        title="Legal Search Engine",
        icon="🔍",  # Magnifying Glass for Search
    )
    project_6_page = st.Page(
        "views/PlaintiffNoticeGeneration.py",
        title="Legal Plaintiff Notice Generator",
        icon="📜",  # Scroll for Legal Documents
    )

    # --- NAVIGATION SETUP [WITH SECTIONS]---
    pg = st.navigation(
        {
            "AI Legal Assistant": [about_page],
            "Tools": [project_1_page, project_2_page, project_3_page, project_4_page, project_5_page, project_6_page],
        }
    )

    # --- RUN NAVIGATION ---
    pg.run()

# Ensure the script is executed only when run directly, not when imported
if __name__ == "__main__":
    main()
