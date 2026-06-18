# External Memory for AI Applications

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](<add Colab URL>)

Workshop notebook for `Virtual Thailand MUG: June 2026 Comunity Online Meetup`

This project demonstrates how to use MongoDB as external memory for an AI application powered by Qwen and LangGraph.

## What You Will Build

The notebook walks through a memory-enabled AI assistant that can:

- call an LLM with basic context
- stream LLM output in the notebook
- save conversation history to MongoDB
- extract durable user profile memory into MongoDB
- call tools with LangGraph
- show each agent loop turn: user message, LLM tool call, tool response, final answer
- summarize runtime context from system prompt, user message, tools, and history
- track token usage per LLM turn and total usage
- save token usage to MongoDB
- run the full chat flow with one command

## Files

- `external_memory_mongodb_langgraph_qwen.ipynb` - main workshop notebook
- `.env.example` - environment variable template
- `.gitignore` - ignores secrets, virtual environments, checkpoints, and local model files

## Main Stack

- LLM: `Qwen/Qwen3-4B-Instruct-2507`
- Agent framework: `langgraph`
- External memory: MongoDB
- LLM adapter: OpenAI-compatible endpoint or local Transformers fallback
- Python libraries: `langgraph`, `langchain`, `langchain-openai`, `pymongo`, `python-dotenv`, `transformers`

## Poster Session

Title: `Memory of AI Application: MongoDB as External Memory for Agentic AI`

Event: `Virtual Thailand MUG: June 2026 Comunity Online Meetup`

Session format: workshop plus poster/demo discussion

### Abstract

Modern AI applications need memory outside the LLM context window. This session demonstrates how MongoDB can act as external memory for an agentic AI application: storing chat history, extracting durable user profile facts, saving working notes, summarizing context, and tracking token usage. The demo uses Qwen, LangGraph, and MongoDB to show a complete memory loop from user message to tool calls, tool responses, persisted memory, and usage telemetry.

### Poster Demo Flow

1. User sends a message to the memory agent.
2. LangGraph routes the message through the LLM.
3. The LLM calls one memory tool per turn.
4. Tools read and write MongoDB collections.
5. The notebook streams LLM output, tool calls, and tool responses.
6. Token usage is summarized per turn and saved to MongoDB.
7. The final answer is produced with history, user profile, tools, and working notes as context.

### Key Takeaways

- LLM context is temporary; application memory should be durable and queryable.
- Conversation history, user profile facts, working notes, summaries, and telemetry should be stored separately.
- Tool calling makes memory explicit and inspectable.
- MongoDB works well as a practical external memory layer for AI applications.
- Token usage should be tracked as part of agent observability.

### Poster Links

Use these placeholders for QR codes or event-share links:

- Notebook: `external_memory_mongodb_langgraph_qwen.ipynb`
- Repository: `<add repository URL>`
- Colab: `<add Colab URL>`
- MongoDB Atlas: `https://www.mongodb.com/products/platform/atlas-database`

Colab URL format for a GitHub-hosted notebook:

```text
https://colab.research.google.com/github/<owner>/<repo>/blob/<branch>/external_memory_mongodb_langgraph_qwen.ipynb
```

## Quick Start in Google Colab

1. Open Google Colab.
2. Upload `external_memory_mongodb_langgraph_qwen.ipynb`.
3. Use a GPU runtime when running the local Transformers fallback:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

4. Run the install cell at the top of the notebook.
5. Create a MongoDB Atlas cluster or use an existing MongoDB connection string.
6. Fill in the `.env` cell values in the notebook.
7. Run the notebook from top to bottom.

For Colab, MongoDB Atlas is usually easier than a local Docker MongoDB server.

## MongoDB Setup

### Option A: MongoDB Atlas

1. Create a free MongoDB Atlas cluster.
2. Create a database user.
3. Allow network access from your notebook environment.
4. Copy your connection string.
5. Set:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/
```

The notebook creates and uses these collections:

- `chat_history`
- `user_personal_information`
- `context_summaries`
- `agent_notes`
- `agent_token_usage`

### Option B: Local MongoDB

Run MongoDB with Docker:

```bash
docker run --name mongodb-memory -p 27017:27017 -d mongo:7
```

Set:

```env
MONGODB_URI=mongodb://localhost:27017
```

If the container already exists:

```bash
docker start mongodb-memory
```

## Environment Variables

Copy `.env.example` to `.env` for local use:

```bash
cp .env.example .env
```

Important values:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=external_memory_demo
MONGODB_HISTORY_COLLECTION=chat_history
MONGODB_PROFILE_COLLECTION=user_personal_information
MONGODB_SUMMARY_COLLECTION=context_summaries
MONGODB_TOKEN_USAGE_COLLECTION=agent_token_usage

OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_REQUEST_TIMEOUT=5
LLM_BACKEND=auto

QWEN_MODEL=Qwen/Qwen3-4B-Instruct-2507
QWEN_TEMPERATURE=0.2
QWEN_MAX_TOKENS=1024

SESSION_ID=demo-session-001
USER_ID=demo-user-001
MODEL_LOCAL_DIR=models/Qwen3-4B-Instruct-2507
```

## LLM Setup

The notebook supports two modes.

### Option A: OpenAI-Compatible Qwen Server

This is the recommended path for stable tool calling.

Example with vLLM:

```bash
pip install -U vllm
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then set:

```env
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
LLM_BACKEND=openai
```

### Option B: Local Transformers Fallback

The notebook can download Qwen weights and run local inference with Transformers.

Set:

```env
LLM_BACKEND=auto
MODEL_LOCAL_DIR=models/Qwen3-4B-Instruct-2507
```

If `OPENAI_API_BASE` is not reachable and the model directory exists, the notebook falls back to local Transformers inference.

## Workshop Flow

Run the notebook sections in order:

1. Install libraries
2. Start or configure MongoDB
3. Download or serve Qwen
4. Create `.env`
5. Load config and connect clients
6. Define MongoDB memory helpers
7. Stream basic LLM output
8. Save chat history
9. Extract user personal information
10. Run LangGraph tool-calling agent loop
11. Inspect MongoDB memory
12. Summarize complete context
13. Review memory design notes
14. Run one command to chat with the agent and save token usage

## One-Command Agent Chat

After running the setup and graph cells, use:

```python
one_command_result = chat_with_memory_agent(
    "Save that I prefer concise bullet-point implementation plans, inspect my memory, and summarize what you know."
)
```

This command:

- streams the agent loop
- prints LLM tool calls
- prints tool responses
- prints token usage per LLM turn
- prints total token usage
- saves token usage to MongoDB

Useful result keys:

```python
one_command_result["run_id"]
one_command_result["token_usage_by_turn"]
one_command_result["token_usage_total"]
one_command_result["token_usage_mongodb_id"]
```

## Memory Model

The notebook separates memory into different MongoDB collections:

- `chat_history` - append-only message logs
- `user_personal_information` - structured durable user facts
- `agent_notes` - non-sensitive working notes saved by tools
- `context_summaries` - compressed summaries of runtime context
- `agent_token_usage` - per-run token usage and tool sequence traces

This separation is important because history, profile memory, working notes, summaries, and usage telemetry have different lifecycles.

## Notes for Workshop Instructors

- Use MongoDB Atlas for the smoothest Colab experience.
- Keep `.env` values private.
- If GPU memory is limited, reduce `QWEN_MAX_TOKENS`.
- If local model loading is slow, run a Qwen-compatible server separately and use `LLM_BACKEND=openai`.
- The notebook intentionally prints each agent turn so participants can see the reasoning loop structure without hiding tool calls.

## Troubleshooting

If `llm.invoke(...)` fails with a connection error:

- make sure your model server is running at `OPENAI_API_BASE`, or
- use `LLM_BACKEND=auto` and run the model download cell first.

If MongoDB connection fails:

- verify `MONGODB_URI`
- check MongoDB Atlas network access
- check username/password encoding in the URI
- confirm the database user has read/write permissions

If Colab runs out of memory:

- use a smaller context length
- reduce `QWEN_MAX_TOKENS`
- prefer an external OpenAI-compatible Qwen server

## License

Workshop material for learning and demonstration.
