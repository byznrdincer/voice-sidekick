"""
Faz 4 — Outlook Calendar MCP
Microsoft Graph API üzerinden (ms-365-mcp-server, npx) Outlook takvimine bağlanır.
--org-mode: kurumsal/iş hesabı özelliklerini (Teams toplantıları dahil) açar.
--preset calendar: sadece takvimle ilgili tool'ları yükler (token tasarrufu).
İlk çalıştırmada terminalde bir link/kod çıkacak, tarayıcıdan giriş yapman gerekecek.
"""

import os
from langchain_mcp_adapters.client import MultiServerMCPClient


def get_calendar_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "calendar": {
                "command": "npx",
                "args": [
                    "-y",
                    "@softeria/ms-365-mcp-server",
                    "--org-mode",
                    "--preset",
                    "calendar",
                ],
                "transport": "stdio",
                "env": {
                    "MS365_MCP_CLIENT_ID": os.getenv("MS_CLIENT_ID", ""),
                    "MS365_MCP_TENANT_ID": os.getenv("MS_TENANT_ID", ""),
                },
            }
        }
    )


async def get_calendar_tools():
    client = get_calendar_mcp_client()
    tools = await client.get_tools()
    return tools
