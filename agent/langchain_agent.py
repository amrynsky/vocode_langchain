
from typing import AsyncGenerator, Optional, Tuple
import logging

from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import AgentConfig, CutOffResponse

from langchain import OpenAI
from langchain.agents import Tool, AgentType
from langchain.tools.ddg_search import DuckDuckGoSearchTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import initialize_agent

class LangchainAgentConfig(AgentConfig, type="langchain_agent"):
    model_name: str = "text-davinci-003"
    temperature: float = 0
    max_tokens: int = 256
    cut_off_response: Optional[CutOffResponse] = None

class LangchainAgent(RespondAgent[LangchainAgentConfig]):
    def __init__(
        self,
        agent_config: LangchainAgentConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(agent_config)
        self.initial_bot_message = (
            agent_config.initial_message.text if agent_config.initial_message else None
        )
        self.logger = logger or logging.getLogger(__name__)
        
        self.llm = OpenAI(  # type: ignore
            model_name=self.agent_config.model_name,
            temperature=self.agent_config.temperature
        )
        
        self.agent = self.create_agent(self.llm)

    def create_agent(
            self,
            llm):
        
        template = """
        Answer the following question as best you can

        Question: {question}
        """

        prompt = PromptTemplate(
            input_variables=["question"],
            template=template
        )
        
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        search = DuckDuckGoSearchTool()

        tools = [
            # FIXME: including LLM as a tool is non-deterministic and causing other tools like Zapier fail
            # it looks like including such tool forces agent to use it not only as a tool
            # 
            # Tool(
            #     name='Language Model',
            #     func=llm_chain.run,
            #     description='use this tool for general purpose questions. Prefer this tool over search for general knowladge and not current events'
            # ),
            Tool(
                name = "Search",
                func=search.run,
                description="useful when you need to answer questions about current events"
            )
        ]

        zapier = ZapierNLAWrapper()
        zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

        tools.extend(zapier_toolkit.get_tools())

        agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
            tools=tools, 
            llm=llm,
            verbose=True
        )

        return agent

    async def respond(
            self,
            human_input,
            conversation_id: str,
            is_interrupt: bool = False,
            ) -> Tuple[str, bool]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
    
        response = await self.agent.arun(human_input)
        
        self.logger.debug(f"LLM response: {response}")
        return response, False

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[str, None]:
        self.logger.debug("LLM generating response to human input")
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield cut_off_response
            return
        
        self.logger.debug("Streaming LLM response")
        yield self.agent.run(human_input)

    def update_last_bot_message_on_cut_off(self, message: str):
        pass