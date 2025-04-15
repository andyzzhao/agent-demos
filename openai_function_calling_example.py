import os
import json
import openai
from typing import Optional
from dotenv import load_dotenv
# Define the calculator tool
def calculator(a: float, b: float, operator: str) -> str:
    """
    Performs basic arithmetic operations on two numbers.
    
    Args:
        a: The first number
        b: The second number
        operator: The arithmetic operator (+, -, *, /)
        
    Returns:
        str: The result of the calculation or an error message
    """
    print(f"Calculator called with: a={a}, b={b}, operator={operator}")
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == '/':
            if b == 0:
                return 'Error: Division by zero'
            return str(a / b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'

# Define the tool schema for the OpenAI API
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Performs basic arithmetic operations on two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number."
                },
                "b": {
                    "type": "number",
                    "description": "The second number."
                },
                "operator": {
                    "type": "string",
                    "description": "The arithmetic operator (+, -, *, /).",
                    "enum": ["+", "-", "*", "/"]
                }
            },
            "required": ["a", "b", "operator"]
        }
    }
}

# Define the available tools
tools = [calculator_tool]

# Define the tool functions mapping
tool_functions = {
    "calculator": calculator,
}

class ToolUseAgent:
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o",
        system_message: str = "You are a helpful assistant that can perform calculations when requested.",
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.system_message = system_message
        self.message_count = 0
        
        # Add system message
        self.add_message("system", system_message)
        
    def add_message(self, role: str, content: str, name=None, tool_call_id=None, tool_calls=None):
        """
        Add a message to the conversation history.
        """
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in tool_calls
            ]
        self.messages.append(message)
        
    def process_tool_calls(self, tool_calls):
        """
        Process tool calls from the assistant.
        """
        tool_results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the appropriate function with the provided arguments
            if function_name in tool_functions:
                result = tool_functions[function_name](**function_args)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
                
        return tool_results
    
    def handle_message(self, user_input: str) -> Optional[str]:
        self.message_count += 1
        
        # Add user message to conversation history
        self.add_message("user", user_input)
        
        # Get assistant's response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Extract the assistant's message
        assistant_message = response.choices[0].message
        
        # Add assistant's message to conversation history
        self.add_message("assistant", assistant_message.content, tool_calls=assistant_message.tool_calls)
        
        # Process tool calls if any
        if assistant_message.tool_calls:
            tool_results = self.process_tool_calls(assistant_message.tool_calls)
            
            # Add tool results to conversation history
            for tool_result in tool_results:
                self.add_message(
                    role="tool",
                    content=tool_result["content"],
                    name=tool_result["name"],
                    tool_call_id=tool_result["tool_call_id"]
                )
            
            # Get final response from assistant
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            
            # Extract and add the final response
            final_content = final_response.choices[0].message.content
            self.add_message("assistant", final_content)
            
            return final_content
                
        return assistant_message.content


def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    agent = ToolUseAgent(
        api_key=api_key,
        system_message="You are a helpful assistant that can perform calculations when requested.",
    )
    
    user_inputs = [
        "Hello",
        "What is 453234.123 * x?",
        "459323.432",
        "Now add 4 and 5 and then devide 3 by 6",
        "What was the first calculation I asked"
    ]
    
    for user_input in user_inputs:
        print(f"# User: '{user_input}'")
        response = agent.handle_message(user_input)
        print(f"# Agent: {response}")

if __name__ == "__main__":
    main()

"""

Class ToolUseAgent:
    def __init__(self, api_key, model, system_message):
        self.api_key = api_key
        self.model = model
        self.system_message = system_message
        self.messages = []
        
    def add_message(self, role: str, content: str, name=None, tool_call_id=None, tool_calls=None):
        # Add message to self.messages

    def process_tool_calls(self, tool_calls):
        # Process tool calls from the assistant


    def handle_message(self, user_input):
        self.add_message("user", user_input)

        # OpenAI ChatCompletion API with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,    # message history
            tools=tools,               # List of Python function definitions
            tool_choice="auto"         # Call zero, one or miltiple functions. 
        )
        
        # Extract the assistant's message
        assistant_message = response.choices[0].message
        
        # Add assistant's message to conversation history
        self.add_message("assistant", assistant_message.content, tool_calls=assistant_message.tool_calls)
        
        # Process tool calls if any
        if assistant_message.tool_calls:
            tool_results = self.process_tool_calls(assistant_message.tool_calls)
            
            # Add tool results to conversation history
            for tool_result in tool_results:
                self.add_message(
                    role="tool",
                    content=tool_result["content"],
                    name=tool_result["name"],
                    tool_call_id=tool_result["tool_call_id"]
                )
            
            # Get final response from assistant
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            
            # Extract and add the final response
            final_content = final_response.choices[0].message.content
            self.add_message("assistant", final_content)
            
            return final_content
                
        return assistant_message.content
"""