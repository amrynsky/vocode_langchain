import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from vocode.streaming.telephony.config_manager.in_memory_config_manager import InMemoryConfigManager
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import InboundCallConfig, TelephonyServer

from agent.agent_factory import AgentFactory

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
            agent_config = ChatGPTAgentConfig(
                initial_message = BaseMessage(text = "Hey Alex! What up?"),
                prompt_preamble = "Have a pleasant conversation about life",
                generate_responses = True,
            ),
        )
    ],
    agent_factory = AgentFactory(),
    logger = logger,
)

app.include_router(telephony_server.get_router())