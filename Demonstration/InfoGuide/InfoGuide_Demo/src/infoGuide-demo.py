import streamlit as st
from PIL import Image
import json
from main import load_qa_pairs, get_response

# Load the JSON file for manufacturing questions
qa_pairs = load_qa_pairs('../assets/qa_pairs.json')

# Load images using PIL
user_icon = Image.open('../assets/manufacturer.png')  # Adjust the path if needed
ai_icon = Image.open('../assets/roboFace.png')

# Inject custom CSS for message styling
st.markdown("""
    <style>
    /* Custom CSS for chat bubble styling */

    /* Flexbox container to hold chat bubbles */
    .chat-container {
        display: flex;
        flex-direction: column;
        overflow-y: auto;
        max-height: 500px; /* Limit height and allow scrolling */
    }

    /* User message bubble */
    .user-bubble {
        background-color: #73000a;
        color: white;
        border: 1px solid #73000a; /* Garnet border */
        border-radius: 15px;
        padding: 10px 20px 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        position: relative;
        float: right;
        clear: both;
    }

    /* AI message bubble */
    .ai-bubble {
        background-color: #f0f0f0;
        color: black;
        border-radius: 15px;
        padding: 10px 10px 10px 20px;
        max-width: 60%;
        margin: 10px 0;
        text-align: left;
        position: relative;
        float: left;
        clear: both;
    }

    /* User icon styling */
    .user-bubble img {
        position: absolute;
        top: -10px;
        right: -50px; /* Adjust position to align it to the right of the message */
        width: 30px;
        height: 30px;
    }

    /* AI icon styling */
    .ai-bubble img {
        position: absolute;
        top: -10px;
        left: -50px; /* Adjust position to align it to the left of the message */
        width: 35px;
        height: 35px;
    }

    /* Ensure space between bubbles */
    .chat-container > div {
        margin-bottom: 20px;
    }

    div.stButton > button {
        background-color: #FFF2E3;
        color: black;
        border-radius: 8px;
        border: black;
        padding: 10px;
        width: 100%;  /* Make the buttons take full width in the columns */
    }

    div.stButton > button:hover {
        background-color: #5a2b27;  /* Slightly darker on hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Add user role selection sidebar
st.sidebar.header("Choose Production Line")
user_roles = ["Rocket Manufacturing", "Vegemite Production", "+ Add New"]
selected_role = st.sidebar.selectbox("", user_roles)

# Define default questions for each production line
default_questions_map = {
    "Rocket Manufacturing": [
        "Will an anomaly occur in the next 10 minutes? If so, what is the nature of the anomaly?",
        "What safety precautions must be taken before operating the MOTOMAN-HC10 manipulator?",
        "How many rockets will be produced in the next hour?"
    ],
    "Vegemite Production": [
        "Will an anomaly occur in the next 10 minutes?",
        "What is the expected demand for Vegemite product type 1 in the next hour?",
        "Provide the inventory status."
    ]
}

# Set default questions based on selected role
if selected_role in default_questions_map:
    default_questions = default_questions_map[selected_role]
else:
    default_questions = ["Will an anomaly occur in the next 10 minutes?", "What are the installation requirements?", "Will production meet demand requirements?"]

# Add some default questions as buttons
selected_question = None

st.write(f"**Related Questions:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(default_questions[0]):
        selected_question = default_questions[0]
with col2:
    if st.button(default_questions[1]):
        selected_question = default_questions[1]
with col3:
    if st.button(default_questions[2]):
        selected_question = default_questions[2]

# User input for the chatbot
user_question = st.chat_input("How may I assist you today?")

# Handle default question button click
if selected_question:
    user_question = selected_question

# Handle user input
if user_question:
    # Append user question to the message history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Get response based on the question
    response = get_response(user_question, qa_pairs)
    
    # Append the assistant's response to the message history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the input field after submission
    user_question = ""  # This line clears the input field

# Display chat messages in a scrollable container
st.write('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # Embed a user icon with a classic image URL
        st.write(f'<div class="user-bubble"><img src="https://cdn-icons-png.flaticon.com/512/747/747545.png" width="10" height="10" />{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # Embed an AI icon with a classic image URL
        st.write(f'<div class="ai-bubble"><img src="https://img.icons8.com/plumpy/24/bot.png" width="10" height="10" />{msg["content"]}</div>', unsafe_allow_html=True)
st.write('</div>', unsafe_allow_html=True)