import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from vocode.streaming.telephony.config_manager.in_memory_config_manager import InMemoryConfigManager
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import InboundCallConfig, TelephonyServer
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, PunctuationEndpointingConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig

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

greeting = "Hey Alex! What's up? How can I help you today?"

chat_gpt_agent_config = ChatGPTAgentConfig(
    initial_message = BaseMessage(text = greeting),
    prompt_preamble = "Have a pleasant conversation about life",
    generate_responses = True,
)

langchain_agent_config = LangchainAgentConfig(
    initial_message=BaseMessage(text = greeting),
    model_name = "text-davinci-003",
    allowed_idle_time_seconds = 60
)

agent_config = langchain_agent_config if os.getenv("AGENT_TYPE") == "LANGCHAIN" else chat_gpt_agent_config

logger.info("Agent: " + agent_config.type)

telephony_server = TelephonyServer(
    base_url = BASE_URL,
    config_manager = config_manager,
    inbound_call_configs = [
        InboundCallConfig(
            url = "/inbound_call",
            agent_config = agent_config,
            synthesizer_config = AzureSynthesizerConfig.from_telephone_output_device(
                # https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=tts
                voice_name = "en-US-MonicaNeural",
            ),
            transcriber_config = DeepgramTranscriberConfig.from_telephone_input_device(
                endpointing_config = PunctuationEndpointingConfig()
            ),
        )
    ],
    agent_factory = AgentFactory(),
    logger = logger,
)

app.include_router(telephony_server.get_router())