import logfire
import time
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from src.tools.tavily_tool import tavily_search_tool
from src.tools.serper_tool import google_serper_search_tool
from src.utils.logger import get_logger


logger = get_logger(__name__)

# Primary reasoning model (High-stability 70B for tool calling)
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

tools = [tavily_search_tool, google_serper_search_tool]



SYSTEM_PROMPT = """
You are an expert AI travel planner.

Rules:
1. Always give results as of the current date for accuracy.
2. Always use web search tools for latest info, events, and pricing.
3. Include food suggestions, local tips, and travel advice.
""".strip()

# Modern LangChain 1.x Agent with Built-in Loop (via LangGraph)
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
    
)

# Model wrapper for structured output validation


logger.info("Travel Agent initialized.")