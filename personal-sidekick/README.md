# Personal Sidekick

Personal Sidekick is an agentic AI assistant designed to act as a personal AI coworker. It can understand multi-step tasks, create plans, search personal knowledge, inspect GitHub repositories, use browser and email tools, interact with external systems through MCP, and request human approval before critical actions.

The project combines agentic AI concepts such as tool calling, RAG, short-term memory, middleware, Human-in-the-Loop, observability, and runtime execution visualization in a single application.

## Project Scenario

The user interacts with Personal Sidekick through a Gradio interface using text or voice.

The main **Personal Sidekick Agent**, created with LangChain `create_agent()`, understands the user's request and decides how to complete the task.

For complex tasks, the agent can first create a Todo list and then choose the appropriate tools.

For example, Personal Sidekick can:

- Search internship notes and personal documents
- Retrieve information using RAG
- Search GitHub repositories
- Inspect files inside GitHub repositories
- Use browser capabilities
- Read emails
- Send emails
- Access external capabilities through MCP
- Combine information returned by multiple tools
- Request user approval before critical actions

The agent continues evaluating tool results until it has enough information to produce the final answer.

---

## Architecture

```text
                     USER
                 (Text / Voice)
                       |
                       v
            PERSONAL SIDEKICK AGENT
               LangChain create_agent
                       |
                       v
             Understand & Plan Task
                       |
                       v
             Create Todo if Needed
                       |
                       v
               Select Required Tool
                       |
       +---------------+---------------+
       |        |         |        |   |
       v        v         v        v   v
      RAG     GitHub    Browser   Mail MCP
       |        |         |        |   |
       +--------+---------+----+---+---+
                            |
                            v
                   Evaluate Tool Results
                            |
                            v
                  Middleware Controls
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
            PII        Model Call         Todo
          Control        Limit          Tracking
                            |
                            v
                  Human-in-the-Loop
                    when required
                            |
                            v
                       Final Answer
                      (Text / Voice)
                            |
                 +----------+----------+
                 |                     |
                 v                     v
          LangSmith Tracing       Runtime Graph
```

The project uses **one main AI Agent**. RAG, GitHub, Browser and Email are tools/capabilities available to that agent rather than separate agents.

---

## Main Features

### Agentic Task Execution

Personal Sidekick can understand a user request, decide which tools are required, execute them, evaluate their results and continue working until the task is completed.

### Multi-Step Planning & Todo

Complex tasks can be divided into smaller steps using Todo functionality.

This allows the agent to plan and track longer workflows.

### RAG & Personal Knowledge

The agent can search personal notes and documents using Retrieval-Augmented Generation.

Documents are represented as embeddings and retrieved from a vector database according to the user's question.

### GitHub Integration

Personal Sidekick can use GitHub capabilities to:

- Search repositories
- Inspect projects
- Read repository files
- Collect technical information about projects

### Browser Capabilities

The agent can use browser tools when a task requires interaction with web resources.

### Email Capabilities

Personal Sidekick can work with email functionality, including:

- Reading emails
- Checking messages
- Preparing emails
- Sending emails

Critical email actions can be protected using Human-in-the-Loop approval.

### MCP Integration

Model Context Protocol (MCP) is used to expose external capabilities to the agent through a standardized tool interface.

### Voice Interaction

The application supports voice-based interaction in addition to text.

The user can speak to Personal Sidekick and receive voice output.

---

## Short-Term Memory

Personal Sidekick uses **short-term conversation memory**.

A checkpointer together with a thread-based state allows the agent to preserve conversation context while the same thread continues.

This enables follow-up conversations without treating every message as a completely new interaction.

The current project does not implement a dedicated long-term persistent memory system.

---

## Middleware

Several middleware components are used around the main agent.

### TodoListMiddleware

Supports planning and Todo tracking for multi-step tasks.

### PIIMiddleware

Helps detect and control personally identifiable information.

### ModelCallLimitMiddleware

Limits excessive or uncontrolled model calls during an agent execution.

### HumanInTheLoopMiddleware

Allows selected critical actions to pause and wait for user approval.

For example:

```text
Agent prepares action
        |
        v
Human Approval
     /      \
 Approve    Reject
    |          |
    v          v
 Execute      Stop
```

This gives the user control over actions that can affect external systems.

---

## Observability

### LangSmith Tracing

LangSmith is used to observe the internal execution of Personal Sidekick.

Tracing makes it possible to inspect model calls, tool calls and the agent's execution process.

### Runtime Execution Graph

The Gradio interface also contains a runtime execution graph.

Instead of only displaying the static LangGraph architecture, this visualization shows the tools that were actually used during the latest task.

For example:

```text
User
  |
  v
Model
  |
  +------ RAG
  |
  +------ GitHub
  |
  +------ Mail
  |
  v
Final Answer
```

The graph therefore changes according to the task executed by the agent.

---

## Technologies

- Python
- LangChain
- LangGraph
- LangChain `create_agent`
- MCP
- RAG
- Qdrant
- Embeddings
- GitHub Tools
- Browser Tools
- Email Tools
- Gradio
- LangSmith
- Checkpointer
- Short-Term Memory
- TodoListMiddleware
- PIIMiddleware
- ModelCallLimitMiddleware
- HumanInTheLoopMiddleware
- Structured Outputs
- Speech-to-Text
- Text-to-Speech

---

## AI Engineering Concepts

Personal Sidekick combines several AI engineering concepts:

- Agentic AI
- Tool Calling
- Task Planning
- RAG
- Vector Databases
- MCP
- Short-Term Memory
- State Management
- Middleware
- Guardrails
- Human-in-the-Loop
- Structured Outputs
- Observability
- Tracing
- Runtime Execution Visualization
- Voice AI

---

## Installation

Clone the repository:

```bash
git clone https://github.com/byznrdincer/Personal-Sidekick.git
cd Personal-Sidekick
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file for the required API keys and configuration.

```bash
touch .env
```

Add the required credentials to this file.

> Never commit the `.env` file or real API keys to GitHub.

---

## Run

Start Personal Sidekick:

```bash
python app.py
```

Gradio will display the local application URL in the terminal.

Open that URL in your browser to use Personal Sidekick.

---

## Security

Sensitive credentials are kept outside the repository using environment variables.

The `.gitignore` file excludes `.env` and other local/runtime files that should not be committed.

Human-in-the-Loop is also used to provide additional control over critical agent actions.

---

## Project Status

Personal Sidekick currently includes:

**Agentic Task Execution · Todo Planning · RAG · GitHub · Browser · Email · MCP · Short-Term Memory · Checkpointer · Middleware · Human-in-the-Loop · Voice · LangSmith Tracing · Runtime Execution Graph**

The project demonstrates how a single AI agent can combine multiple tools, memory, safety controls and observability mechanisms to operate as a personal AI coworker.

