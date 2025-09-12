import logging
import os
from typing import Dict, List, Any

from fastmcp import FastMCP
from openai import OpenAI

# NEW: Pinecone
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ENV ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "poc-discovered-data-query")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "poc")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")  # 3072-dim
INDEX_DIM = int(os.environ.get("INDEX_DIM", "3072"))  # keep consistent with your Pinecone index

# Clients
openai_client = OpenAI()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

server_instructions = """
This MCP server provides semantic search and retrieval over a Pinecone index.
Use 'search' to find relevant items; use 'fetch' to retrieve metadata for a given id.
"""

def _embed_query(q: str) -> List[float]:
    r = openai_client.embeddings.create(model=EMBED_MODEL, input=q)
    v = r.data[0].embedding
    if len(v) != INDEX_DIM:
        raise ValueError(f"Embedding len {len(v)} != INDEX_DIM {INDEX_DIM}. "
                         f"Check model vs. index dimension.")
    return v

def create_server():
    mcp = FastMCP(name="Discovered Pinecone MCP", instructions=server_instructions, stateless_http=True)

    @mcp.tool()
    async def search(query: str, top_k: int = 100, resource_type: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic search over Pinecone.

        Args:
          query: natural language query
          top_k: number of results
          resource_type: optional filter: 'job' | 'candidate' | 'job_activity' | 'candidate_activity'
        """
        if not query or not query.strip():
            return {"results": []}

        # Build embedding
        qvec = _embed_query(query)

        # Optional metadata filter
        pine_filter = None
        if resource_type:
            pine_filter = {"resource_type": {"$eq": resource_type}}

        # Query Pinecone
        res = index.query(
            namespace=PINECONE_NAMESPACE,
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            filter=pine_filter
        )

        out = []
        for m in res.get("matches", []):
            meta = m.get("metadata", {}) or {}
            # Title/snippet come from your ingestion metadata (if present)
            title = meta.get("title") or meta.get("resource_type") or "Result"
            snippet = meta.get("snippet") or ""  # only if you stored it
            url = meta.get("body_url") or ""     # only if you stored a URL
            out.append({
                "id": m["id"],
                "title": title,
                "text": snippet or f"(metadata: {meta})",
                "score": m.get("score"),
                "url": url
            })

        logger.info(f"Pinecone search returned {len(out)} results")
        return {"results": out}

    @mcp.tool()
    async def fetch(id: str) -> Dict[str, Any]:
        """
        Fetch a single vector by id from Pinecone (returns metadata).
        Note: Pinecone doesn't store full bodies unless you include them in metadata.
        """
        if not id:
            raise ValueError("id is required")

        fetched = index.fetch(ids=[id], namespace=PINECONE_NAMESPACE)
        vectors = fetched.get("vectors") or {}
        item = vectors.get(id)
        if not item:
            raise ValueError(f"No item found for id={id}")

        meta = item.get("metadata", {}) or {}
        title = meta.get("title") or id
        url = meta.get("body_url") or ""  # use if you stored source URL

        # If you stored snippets in metadata, return them as 'text'; otherwise show metadata only
        text = meta.get("snippet") or "(no snippet stored; see metadata)"

        return {
            "id": id,
            "title": title,
            "text": text,
            "url": url,
            "metadata": meta
        }

    return mcp

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required (for query embeddings)")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required")
    logger.info(f"Using Pinecone index '{PINECONE_INDEX}' namespace '{PINECONE_NAMESPACE}'")

    # (Optional) sanity check index dimension
    desc = pc.describe_index(PINECONE_INDEX)
    idx_dim = getattr(desc, "dimension", None) or desc["dimension"]
    if idx_dim != INDEX_DIM:
        raise ValueError(f"Index dimension {idx_dim} != INDEX_DIM {INDEX_DIM}. "
                         f"Update INDEX_DIM or recreate the index with the right size.")

    server = create_server()
    server.run(transport="http", port=8000)

if __name__ == "__main__":
    main()