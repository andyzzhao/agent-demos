import asyncio
import os
from typing import Annotated

from semantic_kernel.agents import AssistantAgentThread, OpenAIAssistantAgent
from semantic_kernel.functions import kernel_function

"""
The following sample demonstrates how to create an OpenAI
assistant using OpenAI. The sample shows how to use a 
Semantic Kernel plugin as part of the OpenAI Assistant.
"""


# Define a calculator plugin for the sample
class CalculatorPlugin:
    """A Calculator Plugin that performs basic arithmetic operations."""

    @kernel_function(description="Performs basic arithmetic operations on two numbers.")
    def calculate(
        self, 
        a: Annotated[float, "The first number."],
        b: Annotated[float, "The second number."],
        operator: Annotated[str, "The arithmetic operator (+, -, *, /)."]
    ) -> Annotated[str, "Returns the result of the calculation or an error message."]:
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


# Simulate a conversation with the agent
USER_INPUTS = [
    "Hello",
    "What is 453234.123 * x?",
    "459323.432",
    "Now add 4 and 5 and then devide 3 by 6",
    "What was the first calculation I asked"
]


async def main():
    # 1. Create the client using OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # 2. Create the assistant on the OpenAI service
    client, model = OpenAIAssistantAgent.setup_resources(
        api_key=api_key,
        ai_model_id="gpt-4o"
    )
    
    definition = await client.beta.assistants.create(
        model=model,  # You can change this to another model if needed
        instructions="Answer questions and perform calculations when requested.",
        name="Calculator-Assistant",
    )

    # 3. Create a Semantic Kernel agent for the OpenAI assistant
    agent = OpenAIAssistantAgent(
        client=client,
        definition=definition,
        plugins=[CalculatorPlugin()], 
    )

    thread: AssistantAgentThread = None

    try:
        for user_input in USER_INPUTS:
            print(f"# User: '{user_input}'")
            # 6. Invoke the agent for the current thread and print the response
            async for response in agent.invoke(messages=user_input, thread=thread):
                print(f"# Agent: {response}")
                thread = response.thread
    finally:
        # 7. Clean up the resources
        await thread.delete() if thread else None
        await agent.client.beta.assistants.delete(assistant_id=agent.id)



if __name__ == "__main__":
    asyncio.run(main())