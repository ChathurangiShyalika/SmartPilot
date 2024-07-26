import re

class AssetLoader:

    @staticmethod
    def get_queries():
        # Only include queries for the Documentation Agent
        documentation_agent_queries = [
            "Describe the setup process for the manufacturing plant.",
            "What are the key operational procedures in the manufacturing plant?",
            "How should one handle equipment maintenance and troubleshooting?",
            "Can you provide a guide on safety protocols in the manufacturing plant?",
            "What are the standard operating procedures for quality control in the plant?"
        ]
        queries = {"Documentation_Agent": documentation_agent_queries}
        return queries

    @staticmethod
    def get_templates():
        # Only include template for the Documentation Agent
        system_templates = {
            "Documentation_Agent": """You assist with documentation related to the manufacturing plant. Your role involves providing comprehensive details about the setup process, operational procedures, equipment maintenance, safety protocols, and quality control procedures."""
        }
        return system_templates

    @staticmethod
    def read_data():
        f = open('/content/assets/Final_txt_document_course.txt')
        f_lines = f.read().splitlines()
        f_str = ''.join([re.sub(r'[^A-Za-z0-9 ]+', '' ,line) for line in f_lines if re.sub(r'[^A-Za-z0-9 ]+', '' ,line)])
        f.close()
        return f_str
