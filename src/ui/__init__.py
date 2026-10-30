"""Local reviewer UI (Streamlit; optional extra ``ui``). Start with ``python -m src.ui``.

The UI is a thin client: every human (HITL) decision is sent to ``src.api_server`` with
the reviewer's bearer token, so the state machine and authentication are the API's. AI
steps call the existing runners through :mod:`src.workflows` and only create proposals
and an audit trail of model calls.
"""
