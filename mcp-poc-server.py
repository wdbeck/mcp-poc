import logging
import os
from typing import Dict, List, Any, Tuple
from heapq import nlargest

from fastmcp import FastMCP
from openai import OpenAI
from pinecone import Pinecone

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- ENV ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "poc-discovered-data-query")

# Legacy single-namespace fallback (kept for backward compatibility)
PINECONE_NAMESPACE_FALLBACK = os.environ.get("PINECONE_NAMESPACE", "").strip()

# New: allowed companies config (comma-separated "id:name" pairs)
# Example: PINECONE_ALLOWED_COMPANIES="113:Acme Corp,200:Beta Inc"
PINECONE_ALLOWED_COMPANIES="113:QATC"
_ALLOWED = os.environ.get("PINECONE_ALLOWED_COMPANIES", PINECONE_ALLOWED_COMPANIES).strip()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")  # 3072-dim
INDEX_DIM = int(os.environ.get("INDEX_DIM", "3072"))  # must match your index

# ---------- Clients ----------
openai_client = OpenAI()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

server_instructions = """
This MCP server provides semantic search and retrieval over a Pinecone index.

- Searches are authorization-aware at the namespace level (per-company).
- If the caller specifies a company_id or company_name, queries are routed to that company's namespace.
- If not, queries fan out across all allowed company namespaces (from env config), and results are merged by score.
- resource_type filter values: 'job' | 'candidate' | 'job_activity_event' | 'candidate_activity_event'
"""

# ---------- Helpers ----------

def _embed_query(q: str) -> List[float]:
    r = openai_client.embeddings.create(model=EMBED_MODEL, input=q)
    v = r.data[0].embedding
    if len(v) != INDEX_DIM:
        raise ValueError(f"Embedding len {len(v)} != INDEX_DIM {INDEX_DIM}. "
                         f"Check model vs. index dimension.")
    return v

def _parse_allowed_companies() -> Dict[int, str]:
    """
    Parse PINECONE_ALLOWED_COMPANIES env string like:
      "113:Acme Corp,200:Beta Inc"
    -> {113: "Acme Corp", 200: "Beta Inc"}
    """
    out: Dict[int, str] = {}
    if not _ALLOWED:
        return out
    for part in _ALLOWED.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            cid_str, name = part.split(":", 1)
            cid_str, name = cid_str.strip(), name.strip()
        else:
            cid_str, name = part.strip(), ""
        try:
            cid = int(cid_str)
        except ValueError:
            logger.warning(f"Skipping invalid company id: {cid_str}")
            continue
        out[cid] = name
    return out

ALLOWED_COMPANIES: Dict[int, str] = _parse_allowed_companies()

def namespace_for_company(company_id: int) -> str:
    return f"company_{int(company_id)}"

def _resolve_company_targets(company_id: int | None, company_name: str | None) -> List[Tuple[int, str]]:
    """
    Return list of (company_id, namespace) to query.
    - If company_id is given and allowed -> [that one]
    - Else if company_name is given -> any allowed companies whose name matches (case-insensitive)
    - Else -> all allowed companies
    - If no allowed companies configured, fall back to legacy single namespace (if provided)
    """
    targets: List[Tuple[int, str]] = []

    # Prioritize id
    if company_id is not None:
        if not ALLOWED_COMPANIES or company_id in ALLOWED_COMPANIES:
            targets.append((company_id, namespace_for_company(company_id)))
            return targets
        else:
            logger.info(f"Requested company_id {company_id} not in allowed set; no targets added.")
            return targets

    # Try name
    if company_name:
        name_l = company_name.strip().lower()
        for cid, nm in ALLOWED_COMPANIES.items():
            if nm and nm.lower() == name_l:
                targets.append((cid, namespace_for_company(cid)))
        if targets:
            return targets
        # If names are not configured yet, allow numeric-like names (rare)
        logger.info(f"No exact name match for '{company_name}' in allowed companies.")

    # No hint: fan out across all allowed
    if ALLOWED_COMPANIES:
        for cid in ALLOWED_COMPANIES.keys():
            targets.append((cid, namespace_for_company(cid)))
        return targets

    # Fallback: single namespace mode (legacy)
    if PINECONE_NAMESPACE_FALLBACK:
        # Use (-1) as a sentinel company id when in legacy mode
        targets.append((-1, PINECONE_NAMESPACE_FALLBACK))
        return targets

    # Nothing configured
    return targets

def _search_one_namespace(ns: str, qvec: List[float], top_k: int, pine_filter: Dict[str, Any] | None):
    return index.query(
        namespace=ns,
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        filter=pine_filter
    )

def _merge_results(all_matches: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    # merge by score (desc), keep top_k
    return nlargest(top_k, all_matches, key=lambda m: m.get("score", 0.0))

# ---------- Server ----------

def create_server():
    mcp = FastMCP(name="Discovered Pinecone MCP", instructions=server_instructions, stateless_http=True)

    @mcp.tool()
    async def search(query: str,
                     top_k: int = 20,
                     resource_type: str = "",
                     company_id: int | None = None,
                     company_name: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic search over Pinecone across one or more company namespaces.

        Args:
          query: natural language query
          top_k: number of results to return (after merging)
          resource_type: optional filter ('job' | 'candidate' | 'job_activity_event' | 'candidate_activity_event')
          company_id: optional company id (routes to namespace 'company_{id}')
          company_name: optional company name (matched against allowed companies; case-insensitive)
        """
        if not query or not query.strip():
            return {"results": []}

        # Embed
        qvec = _embed_query(query)

        # Resolve targets (namespaces)
        targets = _resolve_company_targets(company_id, company_name)
        if not targets:
            logger.warning("No company targets resolved. "
                           "Set PINECONE_ALLOWED_COMPANIES or pass company_id/company_name.")
            return {"results": []}

        # Build filter
        pine_filter = None
        if resource_type:
            pine_filter = {"resource_type": {"$eq": resource_type}}

        # Query each namespace; gather all matches
        all_matches: List[Dict[str, Any]] = []
        for cid, ns in targets:
            try:
                res = _search_one_namespace(ns, qvec, top_k, pine_filter)
            except Exception as e:
                logger.exception(f"Query failed for namespace '{ns}': {e}")
                continue
            all_matches.extend(res.get("matches", []))

        # Merge and trim
        merged = _merge_results(all_matches, top_k)

        out = []
        for m in merged:
            meta = m.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("resource_type") or "Result"
            snippet = meta.get("snippet") or ""  # we store short readable text at ingestion
            url = meta.get("body_url") or ""     # if you store URLs
            out.append({
                "id": m["id"],
                "title": title,
                "text": snippet or f"(metadata: {meta})",
                "score": m.get("score"),
                "url": url,
                "company_id": meta.get("company_id"),
                "company_name": meta.get("company_name"),
                "resource_type": meta.get("resource_type"),
                "job_id": meta.get("company_jobposting_id"),
                "candidate_id": meta.get("company_jobposting_candidate_id"),
            })

        logger.info(f"Pinecone search merged {len(all_matches)} matches -> {len(out)} results")
        return {"results": out}

    @mcp.tool()
    async def fetch(id: str,
                    company_id: int | None = None,
                    company_name: str | None = None) -> Dict[str, Any]:
        """
        Fetch a single vector by id from Pinecone (returns metadata).
        If company_id/company_name not provided, searches across allowed namespaces and
        returns the first match found.
        """
        if not id:
            raise ValueError("id is required")

        targets = _resolve_company_targets(company_id, company_name)
        if not targets:
            # fallback to legacy single namespace if configured
            if PINECONE_NAMESPACE_FALLBACK:
                targets = [(-1, PINECONE_NAMESPACE_FALLBACK)]
            else:
                raise ValueError("No company targets resolved for fetch().")

        for _, ns in targets:
            fetched = index.fetch(ids=[id], namespace=ns)
            vectors = fetched.get("vectors") or {}
            item = vectors.get(id)
            if not item:
                continue
            meta = item.get("metadata", {}) or {}
            title = meta.get("title") or id
            url = meta.get("body_url") or ""  # if you stored source URL
            text = meta.get("snippet") or "(no snippet stored; see metadata)"
            return {
                "id": id,
                "title": title,
                "text": text,
                "url": url,
                "metadata": meta
            }

        raise ValueError(f"No item found for id={id} in any allowed namespace.")

    @mcp.tool()
    async def list_allowed_companies() -> Dict[str, Any]:
        """
        Return the companies this MCP is configured to search (from env).
        Useful for debugging and for LLMs to choose a company by name.
        """
        return {
            "companies": [
                {"company_id": cid, "company_name": nm, "namespace": namespace_for_company(cid)}
                for cid, nm in ALLOWED_COMPANIES.items()
            ],
            "legacy_namespace": PINECONE_NAMESPACE_FALLBACK or None
        }

    return mcp

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required (for query embeddings)")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required")
    logger.info(f"Using Pinecone index '{PINECONE_INDEX}'")

    # Optional sanity check: dimension matches embed model
    desc = pc.describe_index(PINECONE_INDEX)
    idx_dim = getattr(desc, "dimension", None) or desc["dimension"]
    if idx_dim != INDEX_DIM:
        raise ValueError(f"Index dimension {idx_dim} != INDEX_DIM {INDEX_DIM}. "
                         f"Update INDEX_DIM or recreate the index with the right size.")

    if ALLOWED_COMPANIES:
        logger.info(f"Allowed companies: {ALLOWED_COMPANIES}")
    elif PINECONE_NAMESPACE_FALLBACK:
        logger.info(f"Legacy single-namespace mode: '{PINECONE_NAMESPACE_FALLBACK}'")
    else:
        logger.warning("No allowed companies configured and no fallback namespace set. "
                       "Set PINECONE_ALLOWED_COMPANIES or PINECONE_NAMESPACE.")

    server = create_server()
    server.run(transport="http", port=8000)

if __name__ == "__main__":
    main()
