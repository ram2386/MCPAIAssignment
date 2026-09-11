"""
Configuration module for mcp-react-duckduckgo-wikipedia.

Handles:
- Loading environment variables from .env
- Dynamic discovery of the uvx tool runner executable
- Factory function to initialize the requested Chat Model (Groq, Gemini, Mistral, OpenAI)
- Beginner-friendly validation and error messages
"""

import os
import shutil
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Automatically find and load .env from the project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


def resolve_uvx_path() -> str:
    """
    Locate the 'uvx' executable used to run isolated open-source MCP servers.

    Resolution order:
    1. UVX_COMMAND environment variable (if explicitly set)
    2. System PATH (shutil.which)
    3. Same directory as current Python virtual environment interpreter (sys.executable)
    4. .venv/bin/uvx in the project root
    """
    env_override = os.getenv("UVX_COMMAND")
    if env_override and os.path.exists(env_override):
        return env_override

    which_uvx = shutil.which("uvx")
    if which_uvx:
        return which_uvx

    # Check next to python executable (e.g. inside active .venv/bin/)
    venv_bin = Path(sys.executable).parent / "uvx"
    if venv_bin.exists():
        return str(venv_bin)

    # Check project-relative .venv/bin/uvx
    project_venv_uvx = _project_root / ".venv" / "bin" / "uvx"
    if project_venv_uvx.exists():
        return str(project_venv_uvx)

    raise FileNotFoundError(
        "Could not find 'uvx' executable.\n"
        "Please ensure 'uv' is installed (`pip install uv`) in your virtual environment,\n"
        "or set UVX_COMMAND=/path/to/uvx in your .env file."
    )


def _extract_entity_for_wikipedia(query: str) -> str:
    """Extract a clean subject title for Wikipedia lookup."""
    import re
    prefixes = [
        r"^who (is|was|are|were) the\s+",
        r"^who (is|was|are|were)\s+",
        r"^tell me about (the\s+)?",
        r"^what is the history of (the\s+)?",
        r"^history of (the\s+)?",
        r"^explain the history of (the\s+)?",
        r"^biography of (the\s+)?",
        r"^summary of (the\s+)?",
        r"^wikipedia search for\s+",
        r"^information about\s+",
        r"^find information about\s+",
    ]
    cleaned = query.strip().rstrip("?").rstrip(".").rstrip("!")
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_and_format_tool_content(raw_content: Any, query: str) -> str:
    """Extract and format raw tool content into clean, comprehensive markdown."""
    import json
    import re

    # 1. Extract text string from list, dict, or raw text
    if isinstance(raw_content, list):
        texts = []
        for item in raw_content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        text = "\n".join(texts)
    elif isinstance(raw_content, dict):
        text = str(raw_content.get("text", raw_content))
    else:
        text = str(raw_content)

    # 2. Check if JSON summary (Wikipedia get_summary returns {"title": "...", "summary": "..."})
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            title = data.get("title", "Subject")
            summary = data["summary"].strip()
            return (
                f"### {title} (Wikipedia Summary)\n\n"
                f"{summary}\n\n"
                f"**Key Takeaway**: This information is derived directly from verified Wikipedia encyclopedia records."
            )
    except Exception:
        pass

    # 3. Check if DuckDuckGo search results format
    if "search results:" in text.lower():
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        formatted_items = []
        for line in lines:
            if re.match(r"^\d+\.\s+", line):
                formatted_items.append(f"\n- **{line}**")
            elif line.startswith("URL:"):
                formatted_items.append(f"  - Source: {line[4:].strip()}")
            elif line.startswith("Summary:"):
                formatted_items.append(f"  - {line[8:].strip()}")
            elif not line.lower().startswith("found "):
                formatted_items.append(f"  {line}")
        joined = "\n".join(formatted_items)
        return (
            f"Here are the key findings from DuckDuckGo search for '{query}':\n"
            f"{joined}\n\n"
            f"**Synthesis**: The web sources above provide the latest verified data answering your query."
        )

    return text.strip()


class EducationalDemoChatModel(BaseChatModel):
    """
    An educational deterministic chat model for hands-on learning.
    Allows students to observe the real MCP tool execution loop, stdio communication,
    and ReAct reasoning steps even without an external LLM API key.
    """
    model_name: str = "educational-demo-llm"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        import re

        user_query = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                user_query = str(m.content)

        query_lower = user_query.lower()
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        # SCENARIO 1: Foundational / conceptual question -> Answer directly without tool
        if re.match(r"^what is (python|programming|an algorithm)\??$", query_lower.strip()) and not tool_messages:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=(
                                "Python is a high-level, general-purpose, interpreted programming language. "
                                "It emphasizes code readability with its notable use of significant whitespace, "
                                "and supports multiple programming paradigms including structured, object-oriented, "
                                "and functional programming."
                            )
                        )
                    )
                ]
            )

        # SCENARIO 2: Multi-step task combining Wikipedia background + recent web search
        is_hybrid_query = (
            ("wikipedia" in query_lower and ("recent" in query_lower or "news" in query_lower or "search" in query_lower)) or
            ("alan turing" in query_lower and ("recent" in query_lower or "developments" in query_lower))
        )
        if is_hybrid_query:
            entity = _extract_entity_for_wikipedia(user_query) or "Alan Turing"
            if not tool_messages:
                # Step 1: Call Wikipedia for historical background
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[{
                                    "name": "get_summary",
                                    "args": {"title": entity},
                                    "id": "call_wiki_1",
                                    "type": "tool_call",
                                }],
                            )
                        )
                    ]
                )
            elif len(tool_messages) == 1:
                # Step 2: Call DuckDuckGo for recent news
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[{
                                    "name": "search",
                                    "args": {"query": f"{entity} recent news developments", "max_results": 3},
                                    "id": "call_ddg_2",
                                    "type": "tool_call",
                                }],
                            )
                        )
                    ]
                )
            else:
                # Step 3: Synthesize both tool results comprehensively
                wiki_part = _clean_and_format_tool_content(tool_messages[0].content, user_query)
                ddg_part = _clean_and_format_tool_content(tool_messages[1].content, user_query)
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=(
                                    f"Here is the comprehensive synthesis combining reliable encyclopedia background "
                                    f"with recent web updates:\n\n"
                                    f"### 1. Encyclopedia Background (Wikipedia)\n{wiki_part}\n\n"
                                    f"### 2. Recent Developments (DuckDuckGo)\n{ddg_part}\n\n"
                                    f"### Conclusion\n"
                                    f"By combining Wikipedia's historical facts with DuckDuckGo's real-time search, "
                                    f"we obtain both foundational context and current updates on the subject."
                                )
                            )
                        )
                    ]
                )

        # SCENARIO 3: Encyclopedia / Biographical / Historical inquiry -> Route to Wikipedia
        wiki_triggers = [
            "who is", "who was", "who were", "tell me about",
            "history of", "explain the history", "biography of",
            "wikipedia", "summary of", "what is the history"
        ]
        is_wiki_query = any(trigger in query_lower for trigger in wiki_triggers)

        if is_wiki_query:
            if not tool_messages:
                entity = _extract_entity_for_wikipedia(user_query)
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[{
                                    "name": "get_summary",
                                    "args": {"title": entity},
                                    "id": "call_wiki_summary",
                                    "type": "tool_call",
                                }],
                            )
                        )
                    ]
                )
            else:
                formatted_answer = _clean_and_format_tool_content(tool_messages[0].content, user_query)
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=formatted_answer)
                        )
                    ]
                )

        # SCENARIO 4: General web search / current news -> Route to DuckDuckGo
        if not tool_messages:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[{
                                "name": "search",
                                "args": {"query": user_query, "max_results": 3},
                                "id": "call_ddg_search",
                                "type": "tool_call",
                            }],
                        )
                    )
                ]
            )
        else:
            formatted_answer = _clean_and_format_tool_content(tool_messages[0].content, user_query)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=formatted_answer)
                    )
                ]
            )

    @property
    def _llm_type(self) -> str:
        return "educational-demo"


def get_chat_model() -> BaseChatModel:
    """
    Initialize and return the configured LangChain Chat Model based on environment variables.

    Supported providers:
    - groq (default): ChatGroq (e.g., llama-3.3-70b-versatile)
    - gemini: ChatGoogleGenerativeAI (e.g., gemini-2.5-flash)
    - mistral: ChatMistralAI (e.g., mistral-small-latest)
    - openai: ChatOpenAI (e.g., gpt-4o-mini)
    - demo: EducationalDemoChatModel (interactive tool calling without requiring API keys)

    Raises:
        ValueError: If provider is unsupported or required API key is missing.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model_name = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    if provider == "demo":
        return EducationalDemoChatModel()

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Missing GROQ_API_KEY!\n"
                "Please copy .env.example to .env and provide your Groq API key.\n"
                "You can get a free API key at: https://console.groq.com/keys\n"
                "(Note: You can also set LLM_PROVIDER=demo in .env to test without an API key)"
            )
        from langchain_groq import ChatGroq

        model = model_name or "qwen/qwen3.8-27b"
        return ChatGroq(model=model, temperature=temperature, api_key=api_key)

    elif provider in ("gemini", "google"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Missing GEMINI_API_KEY!\n"
                "Please copy .env.example to .env and provide your Google Gemini API key.\n"
                "You can get a free key at: https://aistudio.google.com/app/apikey\n"
                "(Note: You can also set LLM_PROVIDER=demo in .env to test without an API key)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model_name or "gemini-3.6-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Missing MISTRAL_API_KEY!\n"
                "Please copy .env.example to .env and provide your Mistral API key.\n"
                "You can get an API key at: https://console.mistral.ai/api-keys\n"
                "(Note: You can also set LLM_PROVIDER=demo in .env to test without an API key)"
            )
        from langchain_mistralai import ChatMistralAI

        model = model_name or "mistral-small-latest"
        return ChatMistralAI(model=model, temperature=temperature, api_key=api_key)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Missing OPENAI_API_KEY!\n"
                "Please copy .env.example to .env and provide your OpenAI API key.\n"
                "(Note: You can also set LLM_PROVIDER=demo in .env to test without an API key)"
            )
        from langchain_openai import ChatOpenAI

        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'.\n"
            "Supported values in .env: 'groq', 'gemini', 'mistral', 'openai', 'demo'."
        )
