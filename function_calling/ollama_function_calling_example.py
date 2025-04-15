import ollama
from typing import List, Optional, Dict, Any, Union

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

# Define the available functions mapping
available_functions = {
    "calculator": calculator,
}

class ToolUseAgent:
    def __init__(
        self, 
        model: str = "llama3",
        system_message: str = "You are a helpful assistant that can perform calculations when requested.",
        max_messages: Optional[int] = None
    ):
        self.model = model
        self.messages = []
        self.system_message = system_message
        self.message_count = 0
        self.max_messages = max_messages
        
        # Add system message
        self.add_message("system", system_message)
        
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
        
        # Get assistant's response using Ollama
        response = ollama.chat(
            self.model,
            messages=self.messages,
            tools=[calculator]  # Pass the actual function reference
        )
        
        # Extract the assistant's message
        assistant_message = response.message
        
        # Handle the case where content might be None
        content = assistant_message.content or ""
        
        # Create a message object for the assistant
        assistant_message_obj = {
            "role": "assistant",
            "content": content
        }
        
        # Add assistant's message to conversation history
        self.messages.append(assistant_message_obj)
        
        # Process tool calls if any
        tool_calls = []
        if hasattr(assistant_message, 'tool_calls'):
            tool_calls = assistant_message.tool_calls or []
            
        if tool_calls:
            for tool_call in tool_calls:
                # Extract function name and arguments based on the structure
                if hasattr(tool_call, 'function'):
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                else:
                    # Handle different possible structures
                    function_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                    function_args = tool_call.get('parameters') if isinstance(tool_call, dict) else getattr(tool_call, 'parameters', {})
                
                # Call the appropriate function with the provided arguments
                if function_name in available_functions:
                    result = available_functions[function_name](**function_args)
                    
                    # Get tool call ID safely
                    tool_call_id = None
                    if hasattr(tool_call, 'id'):
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict) and 'id' in tool_call:
                        tool_call_id = tool_call['id']
                    
                    # Add tool result to conversation history
                    self.add_message(
                        role="tool",
                        content=result,
                        name=function_name,
                        tool_call_id=tool_call_id
                    )
            
            # Get final response from assistant
            final_response = ollama.chat(
                self.model,
                messages=self.messages
            )
            
            # Extract and add the final response
            final_content = ""
            if hasattr(final_response, 'message') and hasattr(final_response.message, 'content'):
                final_content = final_response.message.content or ""
            elif isinstance(final_response, dict) and 'message' in final_response:
                final_content = final_response['message'].get('content', "") or ""
                
            self.add_message("assistant", final_content)
            
            return final_content
        
        return content
    
    def chat(self, user_input: str) -> str:
        response = self.handle_message(user_input)
        if response is None:
            return "Conversation terminated."
        return response

def main():
    agent = ToolUseAgent(
        model="yasserrmd/Qwen2.5-7B-Instruct-1M",
        system_message="You are a helpful assistant that can perform calculations when requested.",
        max_messages=10
    )
    
    # Simulate a conversation with the agent
    user_inputs = [
        "What is 453234.123 * 459323.432?",
    ]
    
    for user_input in user_inputs:
        print(f"# User: '{user_input}'")
        response = agent.chat(user_input)
        print(f"# Agent: {response}")

if __name__ == "__main__":
    main()
