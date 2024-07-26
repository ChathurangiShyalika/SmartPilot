import os
from random import choice
from DataUtils import AssetLoader
from Memory_Utils import Knowledge_Representation, Retr, Symbolic_Model
from Agents import LLM

class MTSS_Copilot:

    @staticmethod
    def simulate_user_turn():
        # Initialize user
        users_and_queries = AssetLoader.get_queries()
        user_roles = list(users_and_queries.keys())
        random_user_role = choice(user_roles)
        random_user_query = choice(users_and_queries[random_user_role])
        return random_user_role, random_user_query

    @staticmethod
    def simulate_QA_agent_turn(user_role, user_query, data):
        llm_response = None
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

            print('\n ===== USER attributes =====\n')
            print('user role:', user_role)
            print('user_query:', user_query)

            agent_instructions, agent_response = MTSS_Copilot.simulate_QA_agent_turn(user_role, user_query, manufacturing_data_repr)
            print('\n ===== SYSTEM INSTRUCTIONS ===== \n', agent_instructions)
            print('\n ===== SYSTEM RESPONSE ===== \n', agent_response)

if __name__ == '__main__':
    MTSS_Copilot.run_demo()
