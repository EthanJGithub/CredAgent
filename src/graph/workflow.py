"""LangGraph workflow — wires the five agents into the decisioning pipeline.

START → ingestion → risk_scoring → policy_compliance → decision
  decision ─(MEDIUM tier / compliance flag, no human yet)→ END (pause for HITL)
  decision ─(otherwise)→ audit → END

Human-in-the-loop: when ``decision`` sets ``requires_human_review`` the graph
ends early. The ``/human-review`` endpoint re-invokes the graph on the same
``thread_id`` with the human decision injected; the MemorySaver checkpointer
restores state and the run proceeds through ``decision`` → ``audit``.
"""
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import CreditDecisionState
from src.agents import (
    ingestion_agent,
    risk_scoring_agent,
    policy_compliance_agent,
    decision_agent,
    audit_agent,
)

logger = logging.getLogger(__name__)

# In-memory checkpointer persists state between the initial run and HITL resume.
memory = MemorySaver()


def ingestion_node(state): return ingestion_agent.run(state)
def risk_scoring_node(state): return risk_scoring_agent.run(state)
def compliance_node(state): return policy_compliance_agent.run(state)
def decision_node(state): return decision_agent.run(state)
def audit_node(state): return audit_agent.run(state)


def route_after_decision(state: CreditDecisionState) -> str:
    if state.get("requires_human_review", False) and state.get("human_decision") is None:
        return "await_human"
    return "audit_agent"


def build_graph():
    graph = StateGraph(CreditDecisionState)

    graph.add_node("ingestion_agent", ingestion_node)
    graph.add_node("risk_scoring_agent", risk_scoring_node)
    graph.add_node("policy_compliance_agent", compliance_node)
    graph.add_node("decision_agent", decision_node)
    graph.add_node("audit_agent", audit_node)

    graph.add_edge(START, "ingestion_agent")
    graph.add_edge("ingestion_agent", "risk_scoring_agent")
    graph.add_edge("risk_scoring_agent", "policy_compliance_agent")
    graph.add_edge("policy_compliance_agent", "decision_agent")
    graph.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {"await_human": END, "audit_agent": "audit_agent"},
    )
    graph.add_edge("audit_agent", END)

    return graph.compile(checkpointer=memory)


app_graph = build_graph()
