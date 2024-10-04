import os
from dotenv import load_dotenv
import json
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from typing_extensions import TypedDict, NotRequired
from typing import Dict
from prompts import reflector_prompt, json_transformer_prompt, reform_researcher_prompt


# Load environment variables from the .env file
load_dotenv()

# Access the environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Quesion Reflector
reflector_llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

reflector_output_parser = StrOutputParser()

reflector_chain = reflector_prompt | reflector_llm | reflector_output_parser


# JSON Transformer
json_transformer_llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    model_kwargs={
        "stream": False,
        "response_format": {"type": "json_object"},
    },
)

json_transformer_output_parser = JsonOutputParser()

json_transformer_chain = (
    json_transformer_prompt | json_transformer_llm | json_transformer_output_parser
)

tool = DuckDuckGoSearchRun(num_results=10, safesearch="off")


def reform_research(question, results):
    reform_researcher_llm = ChatGroq(model="llama-3.2-90b-text-preview", temperature=1)

    reform_researcher_output_parser = StrOutputParser()

    chain = (
        reform_researcher_prompt
        | reform_researcher_llm
        | reform_researcher_output_parser
    )

    response = chain.invoke({"question": question, "results": results})

    return response


def search_insights(questions: dict):
    question_no = 0
    research_result = "---"
    for key, question in questions.items():
        question_no += 1
        if question_no == 3:
            break
        # time.sleep(10)
        results = tool.invoke(question)
        reformed_results = reform_research(question, results)
        research_result += (
            "\n" + f">{key} : {question}" + "\n\n" + reformed_results + "\n" + "---"
        )
    return research_result


class GraphState(TypedDict):
    product: str
    region: str
    target_market: NotRequired[str]
    competitors: NotRequired[str]
    pricing_strategy: NotRequired[str]
    context: NotRequired[str]
    questions_generated: str
    questions_formatted: Dict[str, str]  # Corrected spelling
    research_result: str


def generate_questions(state):
    print("---QUESTIONNING---")
    product = state["product"]
    region = state["region"]
    target_market = state.get("target_market", "")
    competitors = state.get("competitors", "")
    pricing_strategy = state.get("pricing_strategy", "")
    context = state.get("context", "")

    questions_response = reflector_chain.invoke(
        {
            "product": product,
            "region": region,
            "target_market": target_market,
            "competitors": competitors,
            "pricing_strategy": pricing_strategy,
            "contextual_info": context,
        }
    )

    return {"questions_generated": questions_response}


def json_formatter(state):
    print("---FORMATTING---")
    questions_generated = state["questions_generated"]

    formatted_questions = json_transformer_chain.invoke(
        {"questions_generated": questions_generated}
    )

    # Check if formatted_questions is already a dictionary
    if isinstance(formatted_questions, dict):
        formatted_questions_dict = formatted_questions
    else:
        # If it's a string, parse it as JSON
        formatted_questions_dict = json.loads(formatted_questions)

    return {
        "questions_formatted": formatted_questions_dict
    }  # Note the correct spelling here


def research(state):
    print("---RESEARCHING---")
    questions = state["questions_formatted"]

    research = search_insights(questions)

    return {"research_result": research}


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("json_formatter", json_formatter)
workflow.add_node("research", research)

# Add edges
workflow.add_edge(START, "generate_questions")
workflow.add_edge("generate_questions", "json_formatter")
workflow.add_edge("json_formatter", "research")
workflow.add_edge("research", END)

# Set the entry point
workflow.set_entry_point("generate_questions")

# Compile
graph = workflow.compile()

# inputs = {
#     "product": "Nothing Phone 2A",
#     "region": "India",
#     "target_market": "1",
#     "competitors": "1",
#     "pricing_strategy": "1",
#     "context": "1",
# }

# for event in graph.stream(inputs, stream_mode="values"):
#     research = event
#     print(event)

# print(research["research_result"])
