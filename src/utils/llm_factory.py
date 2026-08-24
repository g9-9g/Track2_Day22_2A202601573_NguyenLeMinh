"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


_GEMINI_RATE_LIMITER = None


def _get_gemini_rate_limiter():
    """Dùng chung rate limiter, chừa biên an toàn cho các Gemini free-tier model."""
    global _GEMINI_RATE_LIMITER
    if _GEMINI_RATE_LIMITER is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        _GEMINI_RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=0.20,
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _GEMINI_RATE_LIMITER


def get_llm(
    provider: str = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    json_mode: bool = False,
    gemini_openai_compat: bool = False,
):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)
        max_tokens   : giới hạn output; evaluator có thể cần JSON dài hơn RAG answers
        json_mode    : buộc Gemini trả JSON hợp lệ cho các evaluator có output parser
        gemini_openai_compat: dùng Gemini OpenAI-compatible HTTP endpoint

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        if gemini_openai_compat:
            from langchain_openai import ChatOpenAI

            class GeminiCompatChatOpenAI(ChatOpenAI):
                """Bỏ reasoning tags mà một số Gemma endpoints ghép vào content."""

                @staticmethod
                def _clean_result(result):
                    for item in result.generations:
                        generations = [item] if hasattr(item, "message") else item
                        for generation in generations:
                            content = generation.message.content
                            if isinstance(content, str):
                                cleaned = re.sub(
                                    r"<thought>.*?</thought>",
                                    "",
                                    content,
                                    flags=re.DOTALL,
                                ).strip()
                                generation.message.content = cleaned
                                generation.text = cleaned
                    return result

                def _generate(self, *args, **kwargs):
                    return self._clean_result(super()._generate(*args, **kwargs))

                async def _agenerate(self, *args, **kwargs):
                    result = await super()._agenerate(*args, **kwargs)
                    return self._clean_result(result)

            kwargs = {
                "model": config.GEMINI_MODEL,
                "api_key": config.GOOGLE_API_KEY,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 300,
                "max_retries": 1,
                "rate_limiter": _get_gemini_rate_limiter(),
            }
            if json_mode:
                kwargs["model_kwargs"] = {
                    "response_format": {"type": "json_object"}
                }
            return GeminiCompatChatOpenAI(**kwargs)

        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            rate_limiter=_get_gemini_rate_limiter(),
            request_timeout=60,
            retries=1,
            max_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
            format="json" if json_mode else None,
            num_ctx=8192,
            num_predict=min(max_tokens, 2048),
            keep_alive="10m",
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )
