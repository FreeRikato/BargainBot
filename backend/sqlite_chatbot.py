import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict  # Add this import to resolve the NameError
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools import DuckDuckGoSearchResults

# Load environment variables from the .env file
load_dotenv()

# Access the environment variable for Groq API
groq_api_key = os.getenv("GROQ_API_KEY")


# Define a class for the state that holds conversation history (i.e., messages)
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Build the graph for the chatbot
graph_builder = StateGraph(State)

# Use a search tool as an example tool (replaceable with any other tool)
tool = DuckDuckGoSearchResults(max_results=10)
tools = [tool]

# Initialize the language model (LLM) with tools
llm = ChatGroq(model="llama-3.1-70b-versatile")
llm_with_tools = llm.bind_tools(tools)


# Define the node function to handle the chatbot conversation
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke({"user_message": state["messages"]})]}


# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Add the tool node to the graph
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Define the edges between the chatbot and tools
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

# Set entry and exit points for the graph
graph_builder.set_entry_point("chatbot")

# Use SQLite as a persistence layer for conversation history
with SqliteSaver.from_conn_string("chat.db") as memory:
    # Compile the graph with SQLite memory saver
    graph = graph_builder.compile(checkpointer=memory)


# Function to stream graph updates in a conversation loop
def stream_graph_updates(user_input: str, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}
    for event in graph.stream({"messages": [("user", user_input)]}, config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# Main loop for interacting with the chatbot
while True:
    try:
        # Get user input
        user_input = input("User: ")

        # Exit conditions
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Thanks for the chat! Goodbye.")
            break

        # Stream the conversation updates with the user input
        stream_graph_updates(user_input)

    except Exception as e:
        print(f"An error occurred: {e}")
        break
