import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from random import choice
from assets.DataUtils import AssetLoader
from copilots.Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from copilots.Agents import LLM

class MTSS_Copilot:

    @staticmethod
    def simulate_user_turn():
        # Initialize user
        users_and_queries = AssetLoader.get_queries()
        user_roles = list(users_and_queries.keys())
        
        # Ensure both roles are tested
        # Modify to select each role in alternate turns
        if hasattr(MTSS_Copilot, "turn_count"):
            MTSS_Copilot.turn_count += 1
        else:
            MTSS_Copilot.turn_count = 0
        
        # Alternate between roles for testing
        if MTSS_Copilot.turn_count % 2 == 0:
            random_user_role = 'Anomaly Prediction and Sensor Values'
        else:
            random_user_role = 'Documentation'
        
        random_user_query = choice(users_and_queries[random_user_role])
        return random_user_role, random_user_query

    @staticmethod
    def load_anomaly_prediction_model():
        model_checkpoint = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
        
        unique_labels = [
            'Normal',
            'NoNose,NoBody2,NoBody1',
            'NoNose,NoBody2',
            'NoBody2,NoBody1',
            'NoNose',
            'NoBody1',
            'NoBody2'
        ]
        id2label = {i: label for i, label in enumerate(unique_labels)}
        label2id = {label: i for i, label in enumerate(unique_labels)}
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, 
            num_labels=len(unique_labels), 
            id2label=id2label, 
            label2id=label2id
        )
        
        # Load PEFT (Parameter Efficient Fine-Tuning) configurations if applicable
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin'])
        model = get_peft_model(model, peft_config)
        
        return tokenizer, model, id2label

    @staticmethod
    def get_anomaly_prediction_context(tokenizer, model, id2label, user_query, time_series_data):
        new_input_series = time_series_data  # Replace with the actual time series data if needed
        new_input_instructions = [user_query] * len(new_input_series)
        new_text_inputs = [f"{series} {instruction}" for series, instruction in zip(new_input_series, new_input_instructions)]

        # Tokenize the inputs
        tokenized_inputs = tokenizer(new_text_inputs, padding=True, truncation=True, return_tensors="pt")

        # Move the inputs to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
        model.to(device)

        # Get predictions from the model
        with torch.no_grad():
            logits = model(**tokenized_inputs).logits

        # Convert logits to predicted labels
        predicted_labels = torch.argmax(logits, axis=1)
        predicted_labels = [id2label[label.item()] for label in predicted_labels]

        return predicted_labels

    @staticmethod
    def simulate_QA_agent_turn(user_role, user_query, data):
        llm_response = None
        tokenizer, model, id2label = MTSS_Copilot.load_anomaly_prediction_model()
        
        if user_role == 'Anomaly Prediction and Sensor Values':
            time_series_data = ["[680. 380. 700]"]  # Replace with actual data
            context = MTSS_Copilot.get_anomaly_prediction_context(tokenizer, model, id2label, user_query, time_series_data)
        else:
            context = Retr.retrieve_context(data, user_query, symb_model=Symbolic_Model(), top_k=1)[0]
        
        system_template = AssetLoader.get_templates()[user_role]
        llm = LLM()
        llm.set_prompt(system_template, user_query, context)
        llm_response = llm.respond_to_prompt()
        return system_template, llm_response

    @staticmethod
    def run_demo(turns=2):
        manufacturing_text_data = AssetLoader.read_data()
        manufacturing_data_repr = Knowledge_Representation.organize_data(manufacturing_text_data)

        for _ in range(turns):
            user_role, user_query = MTSS_Copilot.simulate_user_turn()

            print('\n ===== USER ATTRIBUTES =====\n')
            print('User role:', user_role)
            print('User query:', user_query)

            agent_instructions, agent_response = MTSS_Copilot.simulate_QA_agent_turn(user_role, user_query, manufacturing_data_repr)
            print('\n ===== SYSTEM INSTRUCTIONS ===== \n', agent_instructions)
            print('\n ===== SYSTEM RESPONSE ===== \n', agent_response)

if __name__ == '__main__':
    MTSS_Copilot.run_demo()
