import asyncio
import logging
import signal
from dotenv import load_dotenv
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer import AzureSynthesizer
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, TimeEndpointingConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.agent import LLMAgentConfig
from vocode.streaming.models.message import BaseMessage
from langchain_agent import LangchainAgent, LangchainAgentConfig


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()
async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(use_default_devices=False)

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=TimeEndpointingConfig(),
            )
        ),
        agent=LangchainAgent(
            LangchainAgentConfig(
                initial_message=BaseMessage(text="Hey Alex. What's up?"),
                prompt_preamble="""You are a helpful AI assistant. Answer questions in 50 words or less.""",
                model_name="text-davinci-003",
                allowed_idle_time_seconds=60
            )
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output)
        ),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: conversation.terminate())
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())