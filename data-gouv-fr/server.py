import logging
import os

from mcp.server.fastmcp import FastMCP, Context

from clients.search import SearchEngineClient
from core import format_chunk, rerank

# Create an MCP server
server_name = "mcp-data-gouv-fr"
mcp = FastMCP(server_name, log_level="WARNING")
# deps:
# - elasticearch
# - jinja2
# - requests

GREEN = "\033[32m"
RESET = "\033[0m"


#
# Pompt
#


@mcp.prompt()
def albert_simple() -> str:
    prompt = "Tu es un agent de l'état qui répond aux questions des utilisateurs des services publiques de l'état Francais de manière claire, concréte et concise."
    return prompt


@mcp.prompt()
def albert_rag(chunks: list[str]) -> str:
    # @TODO
    # Load and format jinja template
    prompt = ""
    return prompt


#
# Ressources
#


@mcp.resource("document://{id}")
def get_document_by_id(id: str) -> str:
    """Get the document from its reference ID."""
    return f"Hello, {id}!"


#
# Tools
#


@mcp.tool()
def search_albert_collections_v0(ctx: Context, query: str) -> str:
    # Using albert-api collection method
    """Search contextual information about the french public services"""
    # how to create the collection/chunk ?
    return query


@mcp.tool()
def search_albert_collections_v1(ctx: Context, query: str, limit: int = 7) -> str:
    # Using Elasticsearch directly and pyalbert chunking method
    """Search contextual information about the french public services"""
    collection_name = "chunks-v6"
    model_embedding = "BAAI/bge-m3"
    _id_name = "hash"

    se_config = dict(
        es_url=os.getenv("ELASTICSEARCH_URL"),
        es_creds=("elastic", os.getenv("ELASTICSEARCH_PASSWORD")),
        model_embedding=model_embedding,
    )
    se_client = SearchEngineClient(**se_config)
    hits = se_client.search(collection_name, query, limit=limit)
    _contexts = [format_chunk(chunk) for chunk in hits]
    # _ids = [x[_id_name] for x in hits]
    # _sources = [x["url"] for x in hits]
    return "\n\n---\n\n".join(_contexts)


@mcp.tool()
def search_albert_collections_v2(ctx: Context, query: str, limit: int = 12) -> str:
    # Using Elasticsearch directly and pyalbert chunking method + reranker
    """Search contextual information about the french public services"""
    collection_name = "chunks-v6"
    model_embedding = "BAAI/bge-m3"
    model_rerank = "BAAI/bge-reranker-v2-m3"
    _id_name = "hash"

    se_config = dict(
        es_url=os.getenv("ELASTICSEARCH_URL"),
        es_creds=("elastic", os.getenv("ELASTICSEARCH_PASSWORD")),
        model_embedding=model_embedding,
    )
    se_client = SearchEngineClient(**se_config)
    hits = se_client.search(collection_name, query, limit=limit)
    _contexts = [format_chunk(chunk) for chunk in hits]
    # _ids = [x[_id_name] for x in hits]
    # _sources = [x["url"] for x in hits]

    _contexts = rerank(query, _contexts, model=model_rerank)[:6]
    return "\n\n---\n\n".join(_contexts)


if __name__ == "__main__":
    logging.warning(f"{GREEN}Running {server_name}{RESET}")
    mcp.run()
