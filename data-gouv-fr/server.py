import logging
import os
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

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
    prompt = "Tu es un agent de l'état Français qui répond en langue française aux questions des usagers des services publiques et citoyens de manière précise, concrète et concise."
    return prompt


@mcp.prompt()
def search_system_prompt() -> str:
    prompt = "Tu recherche des données précises et à jour en utilisant les outils mis à disposition sur les services publiques français. Réponds de manière précise, conrète et concise."
    return prompt


#
# Ressources
#


@mcp.resource("document://{id}")
def get_document_by_id(id: str) -> str:
    """Get the document from its reference ID."""
    # TODO
    return f"Hello, {id}!"


#
# Tools
#


@mcp.tool()
def search_albert_collections_v0(
    ctx: Context,
    query: Annotated[str, Field(description="A curated, precise question from the original user input.")],
) -> str:
    # Using albert-api collection method
    """Search contextual information about the french public services"""
    # how to create the collection/chunk ?
    return query


@mcp.tool()
def search_albert_collections_v1(
    ctx: Context,
    query: Annotated[
        str, Field(description="Une question précise formulé à partir de l'entrée originale de l'utilisateur.")
    ],
    topics: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Thème de la question (passeport, carte d'identité, retraites, rendez-vous, amendes et infractions, la loi, démarches en ligne, gouvernement, majeur ou mineur, les territoires etc)",
        ),
    ],
) -> str:
    # Using Elasticsearch directly and pyalbert chunking method
    """Recherche de données précises et à jour pour les question concernant les services publiques français (les droits et devoir du citoyens, les lois, assurances, PV et amendes, carte d'identité et formulaire, etc)"""
    collection_name = "chunks-v6"
    model_embedding = "BAAI/bge-m3"
    #collection_name = "chunks-v13-04-25"
    #model_embedding = "bge-multilingual-gemma2"
    _id_name = "hash"
    limit = 10
    q = query + f" ({topics})"

    se_config = dict(
        es_url=os.getenv("ELASTICSEARCH_URL"),
        es_creds=("elastic", os.getenv("ELASTICSEARCH_PASSWORD")),
        model_embedding=model_embedding,
    )
    se_client = SearchEngineClient(**se_config)
    hits = se_client.search(collection_name, q, limit=limit)
    _contexts = [format_chunk(chunk) for chunk in hits]
    # _ids = [x[_id_name] for x in hits]
    # _sources = [x["url"] for x in hits]
    context = "\n\n\n---\n\n\n".join(_contexts)
    context += "\n\n\nPS: To mention relevant contexts, only use the following markdown format: [text related to a context](URL of the context)"
    return context


@mcp.tool()
def search_albert_collections_v2(
    ctx: Context,
    query: Annotated[
        str, Field(description="Une question précise formulé à partir de l'entrée originale de l'utilisateur.")
    ],
    topics: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Thème de la question (passeport, carte d'identité, retraites, rendez-vous, amendes et infractions, la loi, démarches en ligne, gouvernement, majeur ou mineur, les territoires etc)",
        ),
    ],
) -> str:
    # Using Elasticsearch directly and pyalbert chunking method + reranker
    """Recherche de données précises et à jour pour les question concernant les services publiques français (les droits et devoir du citoyens, les lois, assurances, PV et amendes, carte d'identité et formulaire, etc)"""
    collection_name = "chunks-v6"
    model_embedding = "BAAI/bge-m3"
    #collection_name = "chunks-v13-04-25"
    #model_embedding = "bge-multilingual-gemma2"
    model_rerank = "BAAI/bge-reranker-v2-m3"
    _id_name = "hash"
    limit = 20
    limit_rerank = 10
    q = query + f" ({topics})"

    se_config = dict(
        es_url=os.getenv("ELASTICSEARCH_URL"),
        es_creds=("elastic", os.getenv("ELASTICSEARCH_PASSWORD")),
        model_embedding=model_embedding,
    )
    se_client = SearchEngineClient(**se_config)
    hits = se_client.search(collection_name, q, limit=limit)
    _contexts = [format_chunk(chunk) for chunk in hits]
    # _ids = [x[_id_name] for x in hits]
    # _sources = [x["url"] for x in hits]

    _contexts = rerank(q, _contexts, model=model_rerank)[:limit_rerank]
    context = "\n\n\n---\n\n\n".join(_contexts)
    context += "\n\n\nPS: To mention relevant contexts, only use the following markdown format: [text related to a context](URL of the context)"
    return context


if __name__ == "__main__":
    logging.warning(f"{GREEN}Running {server_name}{RESET}")
    mcp.run()
