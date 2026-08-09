import json
import os

import dotenv
from anthropic import Anthropic
from langsmith.evaluation import evaluate
from pydantic import BaseModel

dotenv.load_dotenv()

# Initialize the OpenAI client
client = Anthropic()

# Dataset name in LangSmith (already uploaded)
dataset_name = "news_dataset_class"

# System prompt for extraction
SYSTEM_PROMPT = """Extract information from the news into a dictionary. 
The dictionary keys are company_name, date_of_transaction, amount, product_service, location. 
Make sure the output is in proper JSON with double quotes around the keys and values. 
For all dates, use the following format: mm-dd-yyyy."""

# class Company(BaseModel):
#     compnent_name :str = None
#     date_of_transaction: str = None
#     amount: str = None
#     product_service: str = None

def make_call_to_llm(input):
    # Extract the input content from the dataset item
    user_content = input["news"] if isinstance(input, dict) else input
    
    # Create the message array for the API call
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    # Call the Claude API
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        #output_format=Company,
        messages = messages
    )
    
    # Extract the output from the response
    output = response.content[0].text

    return {"output": output}

# llm_result is the result of the make_call_to_llm call
# It contains the model's response that we'll compare against the expected output
def perform_eval(llm_result, dataset_item):
    try:
        # Parse the model's output
        llm_output = json.loads(llm_result.outputs['output'])
        
        # Parse the expected output
        expected_output = json.loads(dataset_item.outputs['output'])
        
        # Extract score from response
        # For a simpler implementation, let's manually calculate the score
        total_keys = len(expected_output)
        correct_keys = sum(1 for key in expected_output if key in llm_output and llm_output[key] == expected_output[key])
        score = correct_keys / total_keys if total_keys > 0 else 0
        
        return {"score": score}
    except json.JSONDecodeError:
        # Handle the case where JSON parsing fails
        return {"score": 0.0}

# Evaluate the target task
# TODO: Implement the evaluate function to run the evaluation
# See https://docs.smith.langchain.com/evaluation for reference and examples
# This should evaluate make_call_to_llm against the dataset_name using perform_eval
# and create an experiment with the prefix "news_extraction_homework"

result = evaluate(
    make_call_to_llm,
    data = dataset_name,
    evaluators=[perform_eval],
    experiment_prefix="news_extraction_homework"
)
print(result)

# if __name__ == "__main__":
#     res = make_call_to_llm({
#         "news": "Green Energy Solutions has partnered with a local government to implement a city-wide recycling initiative. The contract was signed on May 5, 2024, with a project value of $200,000. This initiative aims to improve waste management and sustainability practices in Austin, TX. This deal is a significant milestone for Green Energy as it aligns with their mission to promote eco-friendly practices."
#     })
#     print(res)