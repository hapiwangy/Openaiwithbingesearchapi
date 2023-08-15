import logging

import azure.functions as func
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.memory.chat_message_histories import CosmosDBChatMessageHistory
from langchain import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

# 設定環境參數
OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "<prefer model version>" # some version may not have function calling ability
OPENAI_API_BASE = "<your Azure endpoint>"
OPENAI_API_KEY = "<your Azure key>"
BING_SUBSCRIPTION_KEY = "<your bing search key>"
BING_SEARCH_URL = "<your bing search url>"


# 設定模型
chat_model = AzureChatOpenAI(openai_api_type=OPENAI_API_TYPE, openai_api_base=OPENAI_API_BASE, openai_api_key=OPENAI_API_KEY, openai_api_version=OPENAI_API_VERSION,deployment_name="gpt-35-turbo")
search = BingSearchAPIWrapper(bing_search_url=BING_SEARCH_URL, bing_subscription_key=BING_SUBSCRIPTION_KEY)
tools = [Tool(
    name = "Search",
    func = search.run,
    description = "useful for when you need to answer questions that you ar not really sure. Such as current events. You should ask targeted questions."
)]

# 建立模板選項

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


# 初始化紀錄的資料庫
## 放上雲端時要根據網路上的資訊做更新
message_history = CosmosDBChatMessageHistory(
  cosmos_endpoint="<your azure cosmos db endpoint>", cosmos_database="<your cosmos db name>", cosmos_container="<your cosmos db container name>", session_id="<your session name>",user_id="<your db id>", connection_string="<your connection string>"
)

message_history.prepare_cosmos()

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

# 建立llmchain以及agent
llm_chain = LLMChain(llm=chat_model, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('content')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('content')

    if name:
        return func.HttpResponse(agent_chain.run(name))
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
