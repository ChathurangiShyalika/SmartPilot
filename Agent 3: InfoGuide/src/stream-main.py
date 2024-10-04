import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from random import choice
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM
import pandas as pd

class MTSS_Copilot:

    @staticmethod
    def load_user_roles():
        users_and_queries = AssetLoader.get_queries()
        return list(users_and_queries.keys()), users_and_queries

    @staticmethod
    def load_anomaly_prediction_model():
        model_checkpoint = 'final_best_model'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        df = pd.read_excel('/Users/ledarssan/AIISC/CCN_models/models/SmartPilot/Agent 3: InfoGuide/src/LLM_FT_dataset.csv')  # Update path as necessary

        unique_labels = df['predicted_label'].unique().tolist()
        id2label = {i: label for i, label in enumerate(unique_labels)}
        label2id = {label: i for i, label in enumerate(unique_labels)}

        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id
        )

        return tokenizer, model, id2label

    @staticmethod
    def get_anomaly_prediction_context(tokenizer, model, id2label, user_query, time_series_data):
        new_input_series = time_series_data
        new_input_instructions = [user_query] * len(new_input_series)
        new_text_inputs = [f"{series} {instruction}" for series, instruction in zip(new_input_series, new_input_instructions)]

        tokenized_inputs = tokenizer(new_text_inputs, padding=True, truncation=True, return_tensors="pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
        model.to(device)

        with torch.no_grad():
            logits = model(**tokenized_inputs).logits

        predicted_labels = torch.argmax(logits, axis=1)
        predicted_labels = [id2label[label.item()] for label in predicted_labels]

        return predicted_labels

    @staticmethod
    def simulate_QA_agent_turn(user_role, user_query, data):
        llm_response = None
        tokenizer, model, id2label = MTSS_Copilot.load_anomaly_prediction_model()

        if user_role == 'Anomaly Prediction and Sensor Values':
            time_series_data = ["[663. 463. 500.]"]
            context = MTSS_Copilot.get_anomaly_prediction_context(tokenizer, model, id2label, user_query, time_series_data)
        else:
            context = Retr.retrieve_context(data, user_query, symb_model=Symbolic_Model(), top_k=1)[0]

        system_template = AssetLoader.get_templates()[user_role]
        llm = LLM()
        llm.set_prompt(system_template, user_query, context)
        llm_response = llm.respond_to_prompt()
        return system_template, llm_response

    @staticmethod
    def run_demo(user_role, user_query):
        manufacturing_text_data = AssetLoader.read_data()
        manufacturing_data_repr = Knowledge_Representation.organize_data(manufacturing_text_data)

        interaction_details = {
            'user_role': user_role,
            'user_query': user_query,
        }

        agent_instructions, agent_response = MTSS_Copilot.simulate_QA_agent_turn(user_role, user_query, manufacturing_data_repr)

        interaction_details['agent_instructions'] = agent_instructions
        interaction_details['agent_response'] = agent_response

        return interaction_details


# Streamlit App
def main():
    st.title("InfoGuide")

    # Sidebar for user role selection
    st.sidebar.header("User Role Selection")
    user_roles, users_and_queries = MTSS_Copilot.load_user_roles()
    selected_role = st.sidebar.selectbox("Choose your role:", user_roles)

    # Display user role description based on selected role
    if selected_role:
        role_descriptions = {
            "Documentation": "You assist users with tasks related to the documentation of the toy rocket manufacturing pipeline. Your responsibilities include: \n \n • Providing detailed instructions on setting up and operating manufacturing machinery. \n \n • Explaining safety protocols and emergency procedures. \n \n • Guiding users through troubleshooting common issues. \n \n • Describing maintenance and calibration procedures. \n \n • Assisting with quality checks and documentation of the production process. \n \n • Ensuring users understand material handling and storage protocols.",
            "Anomaly Prediction and Sensor Values": "You assist users with tasks related to predicting anomalies and determining sensor values in the toy rocket manufacturing pipeline. Your responsibilities include: \n \n • Predicting anomalies that may occur in the next time step based on current and past data. \n \n • Providing the sensor values associated with the next time step. \n \n • Assisting users in understanding the implications of predicted anomalies. \n \n • Offering guidance on how to adjust the manufacturing process to avoid or mitigate predicted anomalies. \n \n • Ensuring that users have the most accurate and up-to-date sensor readings for decision-making."
        }

        role_description = role_descriptions.get(selected_role, "No description available for this role.")
        st.sidebar.markdown(f"**Role Description:** {role_description}")

    # Chat interface with improved bubble styling
    user_query = st.text_input("Type your query:")
    
    if st.button("Submit"):
        if user_query:
            interaction = MTSS_Copilot.run_demo(selected_role, user_query)

            # Inject custom CSS for chat bubble styling and images
            st.markdown("""
                <style>
                /* Chat bubble container */
                .chat-container {
                    display: flex;
                    flex-direction: column;
                    overflow-y: auto;
                    max-height: 500px;
                }

                /* User message bubble */
                .user-bubble {
                    background-color: #73000a;
                    color: white;
                    border: 1px solid #73000a;
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

                /* User profile image */
                .user-bubble img {
                    position: absolute;
                    top: -10px;
                    right: -50px;
                    width: 30px;
                    height: 30px;
                }

                /* AI profile image */
                .ai-bubble img {
                    position: absolute;
                    top: -10px;
                    left: -50px;
                    width: 50px;
                    height: 50px;
                }

                /* Ensure space between bubbles */
                .chat-container > div {
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            # Display user and agent messages in styled bubbles
            st.write('<div class="chat-container">', unsafe_allow_html=True)
            
            # User message with profile picture
            st.markdown(
                f"<div class='user-bubble'><img src='https://cdn-icons-png.flaticon.com/512/747/747545.png' />{interaction['user_query']}</div>",
                unsafe_allow_html=True
            )
            
            # Agent message with profile picture
            st.markdown(
                f"<div class='ai-bubble'><img src='https://cdn-icons-png.freepik.com/512/6783/6783338.png' /><strong>{interaction['user_role']}</strong>: {interaction['agent_response']}</div>",
                unsafe_allow_html=True
            )
            
            st.write('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a query before submitting.")

if __name__ == "__main__":
    main()
