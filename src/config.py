from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # LLM
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "deepseek/deepseek-chat"

    # GitHub
    github_token: str = ""
    github_webhook_secret: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "codebase"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "code-review-agent"

    # API
    api_key: str = "dev-key-change-in-prod"
    max_lines_per_pr: int = 1000
    max_tokens_budget: int = 50_000
    max_iterations: int = 10


settings = Settings()