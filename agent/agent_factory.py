import logging
from typing import Optional
import typing
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import AgentConfig, AgentType, ChatGPTAgentConfig
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.factory import AgentFactory
from agent import LangchainAgentConfig, LangchainAgent

class AgentFactory(AgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if agent_config.type == AgentType.CHAT_GPT:
            return ChatGPTAgent(
                agent_config = typing.cast(ChatGPTAgentConfig, agent_config)
            )
        elif agent_config.type == "langchain_agent":
            return LangchainAgent(
                agent_config = typing.cast(LangchainAgentConfig, agent_config)
            )
        raise Exception("Invalid agent config")