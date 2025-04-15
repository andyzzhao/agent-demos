import os
import json
import openai
import re
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

def calculator(a: float, b: float, operator: str) -> str:
    """
    Performs basic arithmetic operations on two numbers.
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


# Define the tool functions mapping
tool_functions = {
    "calculator": calculator,
}

# Define tool descriptions for the system message
tool_descriptions = {
    "calculator": {
        "description": "Performs basic arithmetic operations on two numbers.",
        "parameters": {
            "a": "The first number (float)",
            "b": "The second number (float)",
            "operator": "The arithmetic operator (+, -, *, /)"
        },
        "example": "calculator(5, 3, '+') returns '8'"
    }
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
        self.system_message = self._create_system_message(system_message)
        self.message_count = 0
        
        # Add system message
        self.add_message("system", self.system_message)
    
    def _create_system_message(self, base_message: str) -> str:
        """
        Create a system message that includes information about available tools.
        
        Args:
            base_message: The base system message
            
        Returns:
            str: The complete system message with tool information
        """
        prompt = "You have access to the following tools:\n\n"
        
        for tool_name, tool_info in tool_descriptions.items():
            prompt += f"- {tool_name}: {tool_info['description']}\n"
            prompt += "  Parameters:\n"
            for param_name, param_desc in tool_info['parameters'].items():
                prompt += f"    - {param_name}: {param_desc}\n"
            prompt += f"  Example: {tool_info['example']}\n\n"
        
        prompt += "To use a tool, format your response as follows:\n"
        prompt += "TOOL_CALL: <tool_name>(<param1>, <param2>, ...)\n"
        prompt += "TOOL_CALL_END\n\n"
        prompt += "After the tool is called, you will receive the result and should continue your response.\n"
        
        return f"{base_message}\n\n{prompt}"
        
    def add_message(self, role: str, content: str, name=None, tool_call_id=None):
        """
        Add a message to the conversation history.
        """
        message = {"role": role, "content": content}
        if name:
            message["name"] = name
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        self.messages.append(message)
    
    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the assistant's message.
        
        Args:
            content: The content of the assistant's message
            
        Returns:
            List of tool calls
        """

        tool_calls = []
        
        # Regular expression to match tool calls
        pattern = r"TOOL_CALL:\s*(\w+)\((.*?)\)\s*TOOL_CALL_END"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = {}
            if args_str.strip():
                # Split by comma but handle quoted strings properly
                arg_parts = []
                current_part = ""
                in_quotes = False
                
                for char in args_str:
                    if char == '"' or char == "'":
                        in_quotes = not in_quotes
                        current_part += char
                    elif char == ',' and not in_quotes:
                        arg_parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part.strip():
                    arg_parts.append(current_part.strip())
                
                # Get parameter names from the tool function
                if tool_name in tool_functions:
                    import inspect
                    sig = inspect.signature(tool_functions[tool_name])
                    param_names = list(sig.parameters.keys())
                    
                    # Map arguments to parameter names
                    for i, arg_part in enumerate(arg_parts):
                        if i < len(param_names):
                            param_name = param_names[i]
                            # Remove quotes if present
                            arg_value = arg_part.strip('"\'')
                            
                            # Convert to appropriate type based on parameter annotation
                            param_type = sig.parameters[param_name].annotation
                            try:
                                if param_type == float:
                                    arg_value = float(arg_value)
                                elif param_type == int:
                                    arg_value = int(arg_value)
                                elif param_type == bool:
                                    arg_value = arg_value.lower() in ('true', '1', 'yes')
                                # More type conversions as needed
                            except ValueError:
                                # If conversion fails, keep as string
                                pass
                                
                            args[param_name] = arg_value
                            print(f"# Parsed arg: {param_name}={arg_value}")
            
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args)
                }
            })
        return tool_calls
    
    def _remove_tool_calls_from_content(self, content: str) -> str:
        """
        Remove tool calls from the content.
        
        Args:
            content: The content of the assistant's message
            
        Returns:
            str: The content with tool calls removed
        """
        pattern = r"TOOL_CALL:.*?TOOL_CALL_END"
        return re.sub(pattern, "", content, flags=re.DOTALL).strip()
        
    def process_tool_calls(self, tool_calls):
        """
        Process tool calls from the assistant.
        
        Args:
            tool_calls: List of tool calls from the assistant
            
        Returns:
            List of tool results
        """
        tool_results = []
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            # Call the appropriate function with the provided arguments
            if function_name in tool_functions:
                result = tool_functions[function_name](**function_args)
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
                
        return tool_results
    
    def handle_message(self, user_input: str) -> Optional[str]:
        """
        Handle a user message and return the assistant's response.
        
        Args:
            user_input: User input message
            
        Returns:
            Assistant's response or None if conversation should terminate
        """
            
        self.message_count += 1
        
        # Add user message to conversation history
        self.add_message("user", user_input)
        
        # Get assistant's response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        # Extract the assistant's message
        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        
        # Parse tool calls from the content
        tool_calls = self._parse_tool_calls(content)
        
        # Remove tool calls from the content
        clean_content = self._remove_tool_calls_from_content(content)
        
        # Create a message object for the assistant
        assistant_message_obj = {
            "role": "assistant",
            "content": clean_content
        }
        
        # Add tool calls if any
        if tool_calls:
            assistant_message_obj["tool_calls"] = tool_calls
        
        # Add assistant's message to conversation history
        self.messages.append(assistant_message_obj)
        
        # Process tool calls if any
        if tool_calls:
            tool_results = self.process_tool_calls(tool_calls)
            
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
        
        return clean_content
    

def main():
    # Get API key from environment variable
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Create the agent with a custom system message and message limit
    agent = ToolUseAgent(
        api_key=api_key,
        system_message="You are a helpful assistant that can perform calculations when requested.",
    )
    
    # Simulate a conversation with the agent
    user_inputs = [
        "Hello",
        "What is 453234.123 * x?",
        "459323.432",
        "Now add 4 and 5 and then devide 3 by 6",
        "What was the first question I asked"
    ]
    
    for user_input in user_inputs:
        print(f"# User: '{user_input}'")
        response = agent.handle_message(user_input)
        print(f"# Agent: {response}")

if __name__ == "__main__":
    main()
