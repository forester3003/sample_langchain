# 検索して答え探してくれる

import os

from dotenv import load_dotenv

from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool

# 「DockDockGo」とは、Google検索のようにウェブ検索が行えるAPIです。難しいAPIキーの発行方法がなく無料で使うことができるツールになります。
# !pip3 install duckduckgo-search
from langchain.tools import DuckDuckGoSearchRun

# APIキー取得
load_dotenv('.env')
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

# 事前に LLM ラッパーを作成しておきます
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 使用するツールをツール名の文字列のリストとして指定します
tool_names = []

# ツールをロードする際に LLM を渡します
tools = load_tools(tool_names, llm=llm)

# 検索を行うツールを定義
search = DuckDuckGoSearchRun()
tools.append(
    Tool(
        name = "duckduckgo-search",
        func = search.run,
        description="useful for when you need to search for latest information in web"
    ),
)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("""
質問内容:
最近リリースされたGoogle社のLLMは？

回答内容:
""")
