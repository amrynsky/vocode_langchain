import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from vocode.streaming.telephony.config_manager.in_memory_config_manager import InMemoryConfigManager
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import InboundCallConfig, TelephonyServer

from agent.agent_factory import AgentFactory
from agent.langchain_agent import LangchainAgentConfig

load_dotenv()

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_manager = InMemoryConfigManager()

BASE_URL = os.getenv("BASE_URL")

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment")

telephony_server = TelephonyServer(
    base_url = BASE_URL,
    config_manager = config_manager,
    inbound_call_configs = [
        InboundCallConfig(
            url = "/inbound_call",
            agent_config = LangchainAgentConfig(
                initial_message=BaseMessage(text="Hey Alex. How can I help you today?"),
                model_name="text-davinci-003",
                allowed_idle_time_seconds=60
            )
        )
    ],
    agent_factory = AgentFactory(),
    logger = logger,
)

app.include_router(telephony_server.get_router())