"""
Faz 4 — GitHub MCP
GitHub'ın resmi remote MCP sunucusuna (streamable HTTP) bağlanıp tool'larını çeker.
"""

import os
from langchain_mcp_adapters.client import MultiServerMCPClient


def get_github_mcp_client() -> MultiServerMCPClient:
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    return MultiServerMCPClient(
        {
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {token}"
                },
            }
        }
    )


async def get_github_tools():
    """GitHub MCP sunucusundan tool listesini async olarak döner."""
    client = get_github_mcp_client()
    tools = await client.get_tools()
    return tools
