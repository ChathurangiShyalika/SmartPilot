import json

# Load JSON file with question-answer pairs
def load_qa_pairs(json_path):
    with open(json_path, 'r') as file:
        qa_pairs = json.load(file)
    return qa_pairs

# Function to process user questions
def get_response(question, qa_pairs):
    # Only check the qa_pairs dictionary based on the user's question
    return qa_pairs.get("general", {}).get(question.lower(), "Sorry, I don't have an answer for that.")

# Example call
if __name__ == "__main__":
    qa_pairs = load_qa_pairs('../assets/qa_pairs.json')
    question = "Is there an anamoly?"  # Example question
    response = get_response(question, qa_pairs)
    print(response)
    
