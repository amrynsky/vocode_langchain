import os
import logging
from dotenv import load_dotenv
from vocode.streaming.telephony.hosted.inbound_call_server import InboundCallServer
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, PunctuationEndpointingConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.agent import ChatGPTAgentConfig

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

def main():
    server = InboundCallServer(
      agent_config=ChatGPTAgentConfig(
        initial_message=BaseMessage(text="Hey Alex! How can I help you today?"),
        prompt_preamble="You are a helpful AI assistant. Answer questions in 50 words or less.",
      ),
      synthesizer_config=AzureSynthesizerConfig.from_telephone_output_device(
          # https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support?tabs=tts
          voice_name="en-US-MonicaNeural",
      ),
      transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
        endpointing_config=PunctuationEndpointingConfig()
      ),
      twilio_config=TwilioConfig(
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
      ),
    )

    server.run(host="0.0.0.0", port=3000)


if __name__ == "__main__":
    main()