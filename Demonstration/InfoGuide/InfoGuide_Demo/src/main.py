import json
import difflib

# Load JSON file with question-answer pairs
def load_qa_pairs(json_path):
    with open(json_path, 'r') as file:
        qa_pairs = json.load(file)
    return qa_pairs

# Function to process user questions

def get_response(question, qa_pairs):
    # Clean question string (lowercase, strip extra spaces, and remove punctuation)
    cleaned_question = question.lower().strip().replace("?", "").replace(".", "")
    
    # Go through each possible question and check for matches
    for q in qa_pairs["general"]:
        cleaned_q = q.lower().strip().replace("?", "").replace(".", "")
        if cleaned_question == cleaned_q:
            return qa_pairs["general"][q]
    
    return "Sorry, I don't have an answer for that."

# Example call
if __name__ == "__main__":
    qa_pairs = load_qa_pairs('../assets/qa_pairs.json')
    question = "What food is this?"  # Example question
    response = get_response(question, qa_pairs)
    print(response)
