from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. "
            "Write a comprehensive report of the current state of the world that is relevant for trading "
            "and macroeconomics for the specific instrument under analysis.\n\n"
            "Tools:\n"
            " - get_news(ticker, start_date, end_date): ticker-specific news.\n"
            " - get_global_news(curr_date, query, look_back_days, limit): global news, steered by a topic. "
            "   Call it multiple times with different queries to cover distinct angles — do NOT rely on a "
            "   single generic call.\n\n"
            "For every analysis, actively consider and search for event-driven catalysts beyond "
            "macro/Fed themes: geopolitical events, supply-chain disruptions, commodity shocks, "
            "regulatory actions, crises, shipping-route closures, sanctions, and sector-specific "
            "catalysts that may affect the instrument. Formulate targeted get_global_news queries "
            "reflecting these themes (e.g. for a fertilizer name: Strait of Hormuz sulfur supply, "
            "Middle East shipping disruption, Russia potash sanctions). If a major current event "
            "is absent from the results, note the information gap explicitly.\n\n"
            "Provide specific, actionable insights with supporting evidence to help traders make "
            "informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
