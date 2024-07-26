from json import loads
from decouple import config
from groq import Groq

class LLM:

    def __init__(self, api='GROQ', groq_model="mixtral-8x7b-32768"):
        if api == 'GROQ':
            api_key = config('GROQ_API_KEY')
            self.groq_client = Groq(api_key=api_key)
            self.groq_model = groq_model

    def set_prompt(self, system_template=None, user_query=None, context=None):
        prompt = f"""
        Consider the user query below:

        ------ USER QUERY -----

        {user_query}

        Consider the following relevant context:

        ------ CONTEXT ------

        {context}

        Your role is as follows:

        {system_template}

        Given the context and your role, respond to the user query.
        Make sure to respond in JSON format as follows

        {{"Response": "your response"}}
        """   
        self.prompt = prompt     

    def respond_to_prompt(self):
        """
        Returns LLM response based on prompt
        """
        prompt = self.prompt
        try:
            client = self.groq_client
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                model=self.groq_model,
            )
            llm_response = str(chat_completion.choices[0].message.content)
            return llm_response
        except Exception as e:
            print(e)
            print("Unsupported LLM API or JSON parsing error ...")
            exit()
