import os
import json
import openai
import re
from typing import Optional, Dict, List, Any, Tuple

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
        max_messages: Optional[int] = None
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.system_message = self._create_system_message(system_message)
        self.message_count = 0
        self.max_messages = max_messages
        
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
        
        Args:
            role: The role of the message sender (user, assistant, system, tool)
            content: The content of the message
            name: The name of the tool (only for tool messages)
            tool_call_id: The ID of the tool call (only for tool messages)
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

        print(f"# Content: {content}")
        tool_calls = []
        
        # Regular expression to match tool calls
        pattern = r"TOOL_CALL:\s*(\w+)\((.*?)\)\s*TOOL_CALL_END"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2)

            print(f"# Tool name: {tool_name}")
            print(f"# Args str: {args_str}")
            
            # Parse arguments
            args = {}
            if args_str.strip():
                # Simple parsing for demonstration - in a real app, you'd need more robust parsing
                arg_pairs = [pair.strip() for pair in args_str.split(',')]
                for pair in arg_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert string values to appropriate types
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                        
                        args[key] = value
            
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args)
                }
            })
        print(f"# Tool calls: {tool_calls}")
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
        # Check for termination conditions
        if "TERMINATE" in user_input:
            return None
            
        # Check message count limit
        if self.max_messages and self.message_count >= self.max_messages:
            return None
            
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
            final_message = final_response.choices[0].message
            final_content = final_message.content or ""
            self.add_message("assistant", final_content)
            
            return final_content
        
        return clean_content
    
    def chat(self, user_input: str) -> str:
        response = self.handle_message(user_input)
        if response is None:
            return "Conversation terminated."
        return response

def main():
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Create the agent with a custom system message and message limit
    agent = ToolUseAgent(
        api_key=api_key,
        system_message="You are a helpful assistant that can perform calculations when requested. If you see 'TERMINATE', stop the conversation.",
        max_messages=10
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
        response = agent.chat(user_input)
        print(f"# Agent: {response}")

if __name__ == "__main__":
    main()
