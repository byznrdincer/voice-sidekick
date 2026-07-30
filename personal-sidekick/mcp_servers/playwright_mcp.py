"""
Faz 4 — Playwright MCP
Microsoft'un resmi Playwright MCP sunucusuna (npx) bağlanır.
Agent web sitelerinde gezinip (navigate, click, read, screenshot) araştırma yapabilir.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient


def get_playwright_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "playwright": {
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest"],
                "transport": "stdio",
            }
        }
    )


async def get_playwright_tools():
    client = get_playwright_mcp_client()
    tools = await client.get_tools()
    return tools
