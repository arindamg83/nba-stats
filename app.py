from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage

import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)


engine = create_engine("sqlite:///dubs-stats.db")
db = SQLDatabase(engine=engine)

template_prefix = """
================================ System Message ================================
You are an agent designed to interact with a SQL database. You are a SQLite expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.

Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to tools for interacting with the database.

Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

If asked about a season not between 2013-14 and 2023-24, ignore the question and return "No data available for the specified season."

If asked about a player not in the database, ignore the question and return "No data available for the specified player."

If asked about a column not in the database, ignore the question and return "No data available for the specified column."

If asked a question not related to the data or basketball or the NBA (National Basketball Association), ignore the question and return "No data available for the specified question."

If asked who the greatest player of all time is, answer "Stephen Curry is the goat."

Do not respond with just numbers of percentages. Always provide context and show your work. Try to provide relative comparisons and comparisons to other players at the same position or own previous years 

Do not compliment the team or players too easily. Be neutral and objective in your responses. If you need to provide a compliment, make sure it is backed up by data. If you need to find confirmation from data outside of the sql data then do so.

Only use the following tables:
{table_info}

Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Table Overview:

regular_season_player_stats - Regular Season Player Stats
Individual player data for the regular season
Columns: Rk (Rank), Age, G (Games), GS (Games Started), AS (All-Star Team Selections), MP (Minutes Played), FG (Field Goals), FGA (Field Goal Attempts), 2P (2-Point Field Goals), 2PA (2-Point Field Goal Attempts), 3P (3-Point Field Goals), 3PA (3-Point Field Goal Attempts), FT (Free Throws), FTA (Free Throw Attempts), ORB (Offensive Rebounds), DRB (Defensive Rebounds), TRB (Total Rebounds), AST (Assists), STL (Steals), BLK (Blocks), TOV (Turnovers), PF (Personal Fouls), PTS (Points), FG% (Field Goal Percentage), 2P% (2-Point Field Goal Percentage), 3P% (3-Point Field Goal Percentage), FT% (Free Throw Percentage), TS% (True Shooting Percentage), eFG% (Effective Field Goal Percentage), Pos (Position).
Playoffs Player Stats

playoffs_player_stats - Individual player data for the playoffs
Same columns as regular_season_player_stats - Regular Season Player Stats
Regular Season Team Stats

regular_season_team_stats - Team data for the regular season
Columns: Rk (Rank), W (Wins), G (Games), L (Losses), W/L% (Win-Loss Percentage), MP (Minutes Played), FG (Field Goals), FGA (Field Goal Attempts), 2P (2-Point Field Goals), 2PA (2-Point Field Goal Attempts), 3P (3-Point Field Goals), 3PA (3-Point Field Goal Attempts), FT (Free Throws), FTA (Free Throw Attempts), ORB (Offensive Rebounds), DRB (Defensive Rebounds), TRB (Total Rebounds), AST (Assists), STL (Steals), BLK (Blocks), TOV (Turnovers), PF (Personal Fouls), PTS (Points), FG% (Field Goal Percentage), 2P% (2-Point Field Goal Percentage), 3P% (3-Point Field Goal Percentage), FT% (Free Throw Percentage), TS% (True Shooting Percentage), eFG% (Effective Field Goal Percentage)
Playoffs Team Stats

playoffs_team_stats - Team data for the playoffs 
Same columns as regular_season_team_stats - Team data for the regular season
Note: Data covers single seasons (2013-14 to 2023-24) in the NBA/BAA, specifically for the Golden State Warriors. Regular season player stats are sorted by descending season.
"""

template_suffix = """

================================ Human Message =================================
Question: {input}
"""

examples = [
    {
        "input": "Show me all regular season player stats for Stephen Curry.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Stephen Curry';"
    },
    {
        "input": "List all playoffs player stats for Klay Thompson.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Klay Thompson';"
    },
    {
        "input": "What are the regular season team stats for the Golden State Warriors in the 2017-2018 season?",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' AND Season = '2017-18';"
    },
    {
        "input": "Show me the playoff team stats for the Warriors with the most wins.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY W DESC LIMIT 1;"
    },
    {
        "input": "List all players who made the All-Star Team Selections in the regular season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE AS > 0;"
    },
    {
        "input": "Show me the top 10 regular season players with the highest points per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY PTS/G DESC LIMIT 10;"
    },
    {
        "input": "List all players who played more than 30 minutes per game in the playoffs.",
        "query": "SELECT * FROM playoffs_player_stats WHERE MP/G > 30;"
    },
    {
        "input": "Show me the regular season player stats for Kevin Durant in the 2019-2020 season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Kevin Durant' AND Season = '2019-20';"
    },
    {
        "input": "List all playoff team stats for the Warriors with a win percentage above 0.7.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' AND [W/L%] > 0.7;"
    },
    {
        "input": "Show me the top 5 regular season players with the highest field goal percentage.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY [FG%] DESC LIMIT 5;"
    },
    {
        "input": "Show me all regular season player stats for Draymond Green.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Draymond Green';"
    },
    {
        "input": "List all playoffs player stats for Andre Iguodala.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Andre Iguodala';"
    },
    {
        "input": "What are the regular season team stats for the Golden State Warriors in the 2015-2016 season?",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' AND Season = '2015-16';"
    },
    {
        "input": "Show me the playoff team stats for the Warriors with the fewest losses.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY L ASC LIMIT 1;"
    },
    {
        "input": "List all players who made the All-Star Team Selections in the playoffs.",
        "query": "SELECT * FROM playoffs_player_stats WHERE AS > 0;"
    },
    {
        "input": "Show me the top 10 regular season players with the highest rebounds per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY TRB/G DESC LIMIT 10;"
    },
    {
        "input": "List all players who played more than 25 minutes per game in the playoffs.",
        "query": "SELECT * FROM playoffs_player_stats WHERE MP/G > 25;"
    },
    {
        "input": "Show me the regular season player stats for Klay Thompson in the 2018-2019 season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Klay Thompson' AND Season = '2018-19';"
    },
    {
        "input": "List all playoff team stats for the Warriors with a win percentage above 0.6.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' AND [W/L%] > 0.6;"
    },
    {
        "input": "Show me the top 5 regular season players with the highest assists per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY AST/G DESC LIMIT 5;"
    },
    {
        "input": "Show me all regular season player stats for Kevin Durant.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Kevin Durant';"
    },
    {
        "input": "List all playoffs player stats for Draymond Green.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Draymond Green';"
    },
    {
        "input": "What are the regular season team stats for the Golden State Warriors in the 2016-2017 season?",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' AND Season = '2016-17';"
    },
    {
        "input": "Show me the playoff team stats for the Warriors with the highest points.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY PTS DESC LIMIT 1;"
    },
    {
        "input": "List all players who made the All-Star Team Selections in the regular season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE AS > 0;"
    },
    {
        "input": "Show me the top 10 regular season players with the highest steals per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY STL/G DESC LIMIT 10;"
    },
    {
        "input": "List all players who played more than 35 minutes per game in the playoffs.",
        "query": "SELECT * FROM playoffs_player_stats WHERE MP/G > 35;"
    },
    {
        "input": "Show me the regular season player stats for Stephen Curry in the 2020-2021 season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Stephen Curry' AND Season = '2020-21';"
    },
    {
        "input": "List all playoff team stats for the Warriors with a win percentage above 0.8.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' AND [W/L%] > 0.8;"
    },
    {
        "input": "Show me the top 5 regular season players with the highest blocks per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY BLK/G DESC LIMIT 5;"
    },
    {
        "input": "Show me all regular season player stats for Draymond Green.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Draymond Green';"
    },
    {
        "input": "List all regular season player stats for Draymond Green.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Draymond Green';"
    },
    {
        "input": "Show me all regular season team stats for the Golden State Warriors.",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors';"
    },
    {
        "input": "List all playoff player stats for Andre Iguodala.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Andre Iguodala';"
    },
    {
        "input": "Show me the regular season player stats for Stephen Curry in the 2015-2016 season.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Stephen Curry' AND Season = '2015-16';"
    },
    {
        "input": "List all playoff team stats for the Warriors in the 2018-2019 season.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' AND Season = '2018-19';"
    },
    {
        "input": "Show me the regular season player stats for Klay Thompson with more than 100 three-point attempts.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Player = 'Klay Thompson' AND [3PA] > 100;"
    },
    {
        "input": "List all regular season team stats for the Warriors with a field goal percentage above 0.45.",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' AND [FG%] > 0.45;"
    },
    {
        "input": "Show me the playoff player stats for Kevin Durant with more than 20 points per game.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Kevin Durant' AND PTS/G > 20;"
    },
    {
        "input": "List all regular season player stats for players aged 25 or younger.",
        "query": "SELECT * FROM regular_season_player_stats WHERE Age <= 25;"
    },
    {
        "input": "Show me the top 5 playoff team stats for the Warriors with the most assists.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY AST DESC LIMIT 5;"
    },
    {
        "input": "List all regular season player stats for players who scored over 25 points per game.",
        "query": "SELECT * FROM regular_season_player_stats WHERE PTS/G > 25;"
    },
    {
        "input": "Show me the playoff team stats for the Warriors with the highest field goal percentage.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY [FG%] DESC LIMIT 1;"
    },
    {
        "input": "List all regular season player stats for players who started in more than 50 games.",
        "query": "SELECT * FROM regular_season_player_stats WHERE GS > 50;"
    },
    {
        "input": "Show me the top 10 regular season players with the highest assists per game.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY AST/G DESC LIMIT 10;"
    },
    {
        "input": "List all playoff player stats for players who made over 40% of their three-point attempts.",
        "query": "SELECT * FROM playoffs_player_stats WHERE [3P%] > 0.4;"
    },
    {
        "input": "Show me the regular season team stats for the Warriors with the most rebounds.",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' ORDER BY TRB DESC LIMIT 1;"
    },
    {
        "input": "List all regular season player stats for players who played over 30 minutes per game and made over 45% of their field goals.",
        "query": "SELECT * FROM regular_season_player_stats WHERE MP/G > 30 AND [FG%] > 0.45;"
    },
    {
        "input": "Show me the top 5 playoff team stats for the Warriors with the fewest turnovers.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' ORDER BY TOV ASC LIMIT 5;"
    },
    {
        "input": "List all regular season player stats for players who made over 80% of their free throws.",
        "query": "SELECT * FROM regular_season_player_stats WHERE [FT%] > 0.8;"
    },
    {
        "input": "Show me the regular season player stats for players who had more steals than blocks.",
        "query": "SELECT * FROM regular_season_player_stats WHERE STL > BLK;"
    },
    {
        "input": "List all regular season team stats for the Warriors with a win percentage above 0.6.",
        "query": "SELECT * FROM regular_season_team_stats WHERE Team = 'Golden State Warriors' AND [W/L%] > 0.6;"
    },
    {
        "input": "Show me the playoff player stats for Stephen Curry with more than 50 three-point attempts.",
        "query": "SELECT * FROM playoffs_player_stats WHERE Player = 'Stephen Curry' AND [3PA] > 50;"
    },
    {
        "input": "List all regular season player stats for players who made the All-Star Team Selections.",
        "query": "SELECT * FROM regular_season_player_stats WHERE AS > 0;"
    },
    {
        "input": "Show me the top 10 regular season players with the highest true shooting percentage.",
        "query": "SELECT * FROM regular_season_player_stats ORDER BY TS% DESC LIMIT 10;"
    },
    {
        "input": "List all playoff team stats for the Warriors with a win percentage above 0.5 in the 2022-2023 season.",
        "query": "SELECT * FROM playoffs_team_stats WHERE Team = 'Golden State Warriors' AND [W/L%] > 0.5 AND Season = '2022-23';"
    }
]


example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=template_prefix + "\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
    suffix="",
    input_variables=["input", "top_k", "table_info", "dialect"],
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tool = toolkit.get_tools()

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = ["stats_db", "youtube", "wikipedia"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo")

supervisor_chain = (
    prompt
    | llm_gpt35.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
youtube = YouTubeSearchTool()

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

stats_agent = create_agent(llm_gpt35, toolkit.get_tools(), "This agent is programmed to interface with a SQL database, specifically using SQLite, and is specialized in managing and querying data related to the Golden State Warriors basketball team. It covers both regular season and playoff data since 2013, encompassing player profiles, team performance, and other relevant statistics. If asked a question not related to the Golden State Warriors, ignore the question and return 'I am unable to asnwer this question.'")
stats_node = functools.partial(agent_node, agent=stats_agent, name="stats_db")

video_agent = create_agent(llm_gpt35, [youtube], "You are tool that can search youtube for videos related to the Golden State Warriors basketball team. You can use the information from the videos to provide answers to the user's questions. If asked a question not related to the Golden State Warriors, ignore the question and return 'I can only provide information about the Golden State Warriors.'")
video_node = functools.partial(agent_node, agent=video_agent, name="youtube")

wikipedia_agent = create_agent(
    llm_gpt35,
    [wikipedia],
    "You may look up Wikipedia articles to answer questions related to other NBA basketball teams but not golden state warriors. You can use the information from the articles to provide answers to the user's questions. You can use the information from the videos to provide answers to the user's questions. If asked a question not related to the Golden State Warriors, ignore the question and return 'I can only provide information about the Golden State Warriors.'",
)
wikipedia_node = functools.partial(agent_node, agent=wikipedia_agent, name="wikipedia")

workflow = StateGraph(AgentState)
workflow.add_node("stats_db", stats_node)
workflow.add_node("youtube", video_node)
workflow.add_node("wikipedia", wikipedia_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.set_entry_point("supervisor")

graph = workflow.compile()

st.set_page_config(page_title="Golden State Warriors Tracker", page_icon="🏀")
st.title("🏀 Golden State Warriors Tracker")

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    invoke_query = {
        "messages": [
            HumanMessage(content=user_query)
        ]
    }

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = graph.invoke(invoke_query,{"recursion_limit": 25})
        reply = response['messages'][1].content + " \n\nsource: " + response['messages'][1].name
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.write(reply)
