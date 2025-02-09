import streamlit as st
import pandas as pd

# Sample data of people registered for a meeting
data = {
    "Name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "David Wilson", "Eva Green"],
    "Email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com", "eva@example.com"],
    "Phone": ["123-456-7890", "987-654-3210", "555-666-7777", "111-222-3333", "444-555-6666"],
    "Appointment Date": ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04", "2024-06-05"],
    "Meeting Time": ["10:00 AM", "11:00 AM", "01:00 PM", "02:00 PM", "03:00 PM"],
    "Meeting Link": [
        "https://meet.jit.si/alice-meet-id",
        "https://meet.jit.si/bob-meet-id",
        "https://meet.jit.si/charlie-meet-id",
        "https://meet.jit.si/david-meet-id",
        "https://meet.jit.si/eva-meet-id"
    ]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Adding a status column to store the accept/reject status
df['Status'] = ['Pending'] * len(df)

def main():
    st.title("Lawyer Appointment Registrations")
    st.write("Below are the details of people registered for a meeting with the lawyer:")

    # Use session state to persist DataFrame changes across interactions
    if 'df' not in st.session_state:
        st.session_state.df = df

    # Display the data in a table with Accept/Reject and Join Meet buttons
    for i, row in st.session_state.df.iterrows():
        st.write(f"### {row['Name']}")
        st.write(f"*Email:* {row['Email']}")
        st.write(f"*Phone:* {row['Phone']}")
        st.write(f"*Appointment Date:* {row['Appointment Date']}")
        st.write(f"*Meeting Time:* {row['Meeting Time']}")
        st.write(f"*Status:* {row['Status']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Accept', key=f'accept_{i}'):
                st.session_state.df.at[i, 'Status'] = 'Accepted'
                st.success(f'Meeting with {row["Name"]} has been scheduled.')
        with col2:
            if st.button('Reject', key=f'reject_{i}'):
                st.session_state.df.drop(i, inplace=True)
                st.session_state.df.reset_index(drop=True, inplace=True)
                st.error(f'Meeting with {row["Name"]} has been rejected.')
                st.experimental_rerun()  # To refresh the table and remove the rejected entry
        with col3:
            meet_link = row['Meeting Link']
            if st.button('Join Meet', key=f'join_{i}'):
                st.session_state.meet_link = meet_link
                st.experimental_rerun()

        st.write("---")  # Divider between rows

    # Embed Jitsi Meet for the selected meeting
    if 'meet_link' in st.session_state:
        st.write("### Join the Meeting")
        st.markdown(f"""
            <iframe src="{st.session_state.meet_link}" style="width: 100%; height: 600px; border: 0;"></iframe>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()