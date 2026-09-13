from google.adk.apps import App
from appealerAgent.agent import root_agent

app = App(
    root_agent=root_agent,
    name="app",
)

__all__ = ["root_agent", "app"]

