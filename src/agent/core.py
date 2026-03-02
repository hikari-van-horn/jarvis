import os
import logging
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL

logger = logging.getLogger("agent")

def load_prompt(agent_name:str, template_name: str, **kwargs) -> str:
    """Loads a prompt template from the prompts directory and formats it."""
    prompt_path = os.path.join(os.path.dirname(__file__), agent_name, template_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
            return template.format(**kwargs)
    except Exception as e:
        logger.error(f"Error loading prompt {template_name}: {e}")
        return "You are a helpful AI assistant."

class DiscordAgentGraph:
    def __init__(self):
        self.memory = MemorySaver()
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY or "dummy_key_if_not_set",
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
            model="deepseek-chat",
            temperature=0.7
        )
        
        # Build the state graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)
        
        self.app = graph_builder.compile(checkpointer=self.memory)

    async def _chatbot_node(self, state: MessagesState):
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

    async def chat(self, user_id: str, user_name: str, user_input: str) -> str:
        """Process user input through the LangGraph app and return response."""
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_or_deepseek_api_key_here":
            return f"⚠️ API key not configured yet.\n\nYour message: {user_input}"
            
        config = {"configurable": {"thread_id": str(user_id)}}
        
        # Check current state for thread history
        state = self.app.get_state(config)
        messages_to_send = []
        
        # If this is the first message in the thread, prepend system prompt
        if not state.values.get("messages"):
            soul_prompt = load_prompt("jarvis", "soul.md")
            sys_prompt_content = load_prompt("jarvis", "system_prompt.md", user_name=user_name, agent_soul=soul_prompt)
            messages_to_send.append(SystemMessage(content=sys_prompt_content))
            
        messages_to_send.append(HumanMessage(content=user_input))
        
        try:
            # Run graph asynchronously
            # Streaming values to get final state or just calling ainvoke
            final_state = await self.app.ainvoke({"messages": messages_to_send}, config)
            # final_state["messages"] contains the conversation so far, return last Ai message
            return final_state["messages"][-1].content
        except Exception as e:
            logger.error(f"Error communicating with LangGraph agent: {e}")
            return f"⚠️ I encountered an error while trying to process your request: {e}"
