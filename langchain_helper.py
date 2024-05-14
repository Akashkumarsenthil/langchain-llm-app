import os
from langchain_openai import OpenAI  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType



load_dotenv()

# Check if the API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY is not set in .env file.")
else:
    print("API Key loaded successfully.")

def generate_pet_name(animal_type, pet_color):
    try:
        llm = OpenAI(temperature=0.7)
        
        prompt_template_name = PromptTemplate(
            input_variables = ['animal_type', 'pet_color'],
            template = "I have a {animal_type} pet, it is {pet_color} in color and I want a cool name. Suggest me five cool names for my pet, formatted as a numbered list with each name on a new line."
        )
    
        name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")
        
        response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})

        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def langchain_agent():
    llm = OpenAI(temperature=0.5)

    tools = load_tools(["wikipedia", "llm-math"], llm = llm)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    result = agent.run(
        "What is the average age of a dog? Multiply the age by 3"
    )

    print(result)



if __name__ == "__main__":
    print(generate_pet_name("Dog", "Black"))
    #print (generate_pet_name())
    #langchain_agent()

#generations=[[Generation(text='\n\n1. Maverick\n2. Luna\n3. Diesel\n4. Koda\n5. Nova', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 34, 'total_tokens': 56}, 'model_name': 'gpt-3.5-turbo-instruct'} run=[RunInfo(run_id=UUID('db89fbb8-b078-40a3-bc11-7d55384f0684'))]







