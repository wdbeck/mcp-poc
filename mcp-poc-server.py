#!/usr/bin/env python3
"""
MCP server: Pinecone semantic search with per-user, per-company, per-job access control.

Quick start (example):

export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX="poc-discovered-data-query"

# Server allowlist of companies (ops-config). Only these companies can be queried by this server.
# Format: "id:name,id:name,..."
export PINECONE_ALLOWED_COMPANIES="113:QATC"

# Single-user access profile for this POC (what THIS user can see).
# role: "admin" | "editor" | "viewer"
# all_jobs: true -> full company access; false -> restrict to the provided job_ids
export DISCOVERED_ACCESS_PROFILE='{
  "user_id": "weston",
  "companies": {
    "113": { "role": "viewer", "all_jobs": false, "job_ids": [1001, 1002] }
  }
}'

# (Optional) legacy single-namespace mode if you don't use company-based namespaces:
# export PINECONE_NAMESPACE="legacy_namespace"

python mcp-poc-server.py
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from heapq import nlargest
from typing import Any, Dict, List, Optional, Set, Tuple, Literal

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

# Allowed companies for THIS SERVER (ops-config). If unset, no server-level allowlist is enforced,
# and the user's access profile alone determines what they can search.
_ALLOWED = os.environ.get("PINECONE_ALLOWED_COMPANIES", "").strip()

EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-large")  # 3072-dim as of writing
INDEX_DIM = int(os.environ.get("INDEX_DIM", "3072"))  # must match your index dimension

# ---------- Clients ----------
openai_client = OpenAI()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

server_instructions = """
This MCP server provides semantic search and retrieval over a Pinecone index.

- Multi-tenant: each company maps to a Pinecone namespace "company_{company_id}".
- Per-user, per-company, per-job access control:
  * Admin  -> all jobs in that company
  * Editor -> either all jobs or a restricted set of job IDs
  * Viewer -> same as Editor for viewing (no write ops in this POC)
- If the caller specifies a company_id or company_name, we restrict to that company (if allowed).
- If not, we fan out across ONLY the companies the user can access (intersected with the server allowlist, if set).
- resource_type filter values: 'job' | 'candidate' | 'job_activity_event' | 'candidate_activity_event'
"""

# ---------- Permissions / Access Model (Phase 1: single-user via env) ----------
Role = Literal["admin", "editor", "viewer"]

@dataclass
class CompanyScope:
    role: Role
    all_jobs: bool = False
    job_ids: Optional[Set[int]] = None  # if all_jobs=False, restrict to these job IDs

@dataclass
class AccessProfile:
    user_id: str
    companies: Dict[int, CompanyScope]  # key = company_id

def load_access_profile() -> AccessProfile:
    """
    Reads a single user's access from env DISCOVERED_ACCESS_PROFILE (JSON).
    Example value:
      {
        "user_id": "weston",
        "companies": {
          "113": { "role": "viewer", "all_jobs": false, "job_ids": [1001, 1002] }
        }
      }
    """
    raw = os.environ.get("DISCOVERED_ACCESS_PROFILE", "").strip()
    if not raw:
        raise ValueError(
            "DISCOVERED_ACCESS_PROFILE is required (JSON). "
            "See docstring in load_access_profile() for format."
        )
    data = json.loads(raw)
    comps: Dict[int, CompanyScope] = {}
    for cid_str, scope in (data.get("companies") or {}).items():
        cid = int(cid_str)
        comps[cid] = CompanyScope(
            role=scope.get("role", "viewer"),
            all_jobs=bool(scope.get("all_jobs", False)),
            job_ids=set(map(int, scope.get("job_ids", []))) or None
        )
    profile = AccessProfile(
        user_id=data.get("user_id", "unknown"),
        companies=comps
    )
    return profile

ACCESS_PROFILE = load_access_profile()

# ---------- Helpers: server allowlist parsing ----------

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

# ---------- Query helpers ----------

async def _embed_query(q: str) -> List[float]:
    # Offload blocking network call so the async tool handler isn't stalled
    def _do():
        r = openai_client.embeddings.create(model=EMBED_MODEL, input=q)
        v = r.data[0].embedding
        if len(v) != INDEX_DIM:
            raise ValueError(
                f"Embedding len {len(v)} != INDEX_DIM {INDEX_DIM}. "
                f"Check model vs. index dimension."
            )
        return v
    return await asyncio.to_thread(_do)

async def _search_one_namespace(ns: str, qvec: List[float], top_k: int,
                                pine_filter: Dict[str, Any] | None):
    def _do():
        return index.query(
            namespace=ns,
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            filter=pine_filter
        )
    return await asyncio.to_thread(_do)

def _merge_results(all_matches: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    # merge by score (desc), keep top_k
    return nlargest(top_k, all_matches, key=lambda m: m.get("score", 0.0))

# ---------- Access enforcement helpers ----------

def _intersection_user_server_allowed() -> Set[int]:
    """Intersect user's companies with server allowlist (if set)."""
    user_company_ids = set(ACCESS_PROFILE.companies.keys())
    if ALLOWED_COMPANIES:
        return user_company_ids & set(ALLOWED_COMPANIES.keys())
    return user_company_ids

def _resolve_company_targets(company_id: int | None, company_name: str | None) -> List[Tuple[int, str]]:
    """
    Return list of (company_id, namespace) to query, intersected with what the user is allowed to see.
    Priority:
      1) company_id if provided & allowed
      2) company_name exact (case-insensitive) match within allowed set
      3) otherwise fan-out across all allowed companies for this user
    """
    targets: List[Tuple[int, str]] = []
    allowed_ids = _intersection_user_server_allowed()
    if not allowed_ids:
        logger.info("Caller has no allowed companies.")
        return targets

    # ID takes priority
    if company_id is not None:
        if company_id in allowed_ids:
            return [(company_id, namespace_for_company(company_id))]
        logger.info(f"Requested company_id {company_id} not allowed for this user.")
        return targets

    # Name lookup (exact, case-insensitive) within allowed set
    if company_name:
        name_l = company_name.strip().lower()
        for cid in allowed_ids:
            nm = ALLOWED_COMPANIES.get(cid, "")
            if nm and nm.lower() == name_l:
                return [(cid, namespace_for_company(cid))]
        logger.info(f"No exact name match for '{company_name}' in user's allowed companies.")

    # No hint: fan out across ALL companies the user can access
    for cid in sorted(allowed_ids):
        targets.append((cid, namespace_for_company(cid)))
    return targets

def _company_filter_for(cid: int, base_filter: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """
    Build a Pinecone filter for this company based on the user's role and job visibility.
    Returns None if the user has no access to this company.
    """
    scope = ACCESS_PROFILE.companies.get(cid)
    if not scope:
        return None  # no access

    f = dict(base_filter or {})
    # Admin or "all jobs" -> no job restriction
    if scope.role == "admin" or scope.all_jobs:
        return f or None

    # Restrict to allowed job ids
    job_ids = scope.job_ids
    if not job_ids:
        # Has company-level entry but zero jobs -> produce no results
        return {"company_jobposting_id": {"$in": []}}

    job_clause = {"company_jobposting_id": {"$in": sorted(map(int, job_ids))}}
    return {"$and": [f, job_clause]} if f else job_clause

def _is_allowed(cid: Any, job_id: Any) -> bool:
    """
    Post-fetch guard: verify the item belongs to a company/job the user can view.
    """
    if cid is None:
        return False
    try:
        cid = int(cid)
    except Exception:
        return False

    scope = ACCESS_PROFILE.companies.get(cid)
    if not scope:
        return False

    if scope.role == "admin" or scope.all_jobs:
        return True

    if job_id is None:
        return False
    try:
        job_id = int(job_id)
    except Exception:
        return False

    return job_id in (scope.job_ids or set())

# ---------- Validation ----------
VALID_RESOURCE_TYPES = {"job", "candidate", "job_activity_event", "candidate_activity_event"}

def _validate_resource_type(resource_type: str) -> None:
    if resource_type and resource_type not in VALID_RESOURCE_TYPES:
        raise ValueError(
            f"Invalid resource_type: {resource_type}. "
            f"Valid: {sorted(VALID_RESOURCE_TYPES)}"
        )

# ---------- Server ----------

def create_server():
    mcp = FastMCP(name="Discovered Pinecone MCP", instructions=server_instructions, stateless_http=True)

    @mcp.tool()
    async def list_allowed_companies() -> Dict[str, Any]:
        """
        Return both the server's allowlist (ops-config) and THIS user's visible companies.
        Helpful for debugging & discovery.
        """
        user_companies = []
        for cid, scope in ACCESS_PROFILE.companies.items():
            user_companies.append({
                "company_id": cid,
                "role": scope.role,
                "all_jobs": scope.all_jobs,
                "job_ids": sorted(scope.job_ids) if scope.job_ids else None,
                "namespace": namespace_for_company(cid),
                "server_allowed": (cid in ALLOWED_COMPANIES) if ALLOWED_COMPANIES else True,
                "server_name": ALLOWED_COMPANIES.get(cid) if ALLOWED_COMPANIES else None
            })

        return {
            "server_allowlist": [
                {"company_id": cid, "company_name": nm, "namespace": namespace_for_company(cid)}
                for cid, nm in ALLOWED_COMPANIES.items()
            ],
            "legacy_namespace": PINECONE_NAMESPACE_FALLBACK or None,
            "user_profile": {
                "user_id": ACCESS_PROFILE.user_id,
                "companies": user_companies
            }
        }

    @mcp.tool()
    async def search(query: str,
                     top_k: int = 20,
                     resource_type: str = "",
                     company_id: int | None = None,
                     company_name: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic search over Pinecone across one or more company namespaces, respecting user permissions.

        Args:
          query: natural language query
          top_k: number of results to return (after merging)
          resource_type: optional filter ('job' | 'candidate' | 'job_activity_event' | 'candidate_activity_event')
          company_id: optional company id (routes to namespace 'company_{id}', if allowed)
          company_name: optional company name (matched against server allowlist names; case-insensitive)
        """
        if not query or not query.strip():
            return {"results": []}

        # Clamp top_k sensibly
        try:
            top_k = max(1, min(int(top_k), 200))
        except Exception:
            top_k = 20

        _validate_resource_type(resource_type)

        # Embed
        qvec = await _embed_query(query)

        # Resolve targets (namespaces) within user/server allowed set
        targets = _resolve_company_targets(company_id, company_name)
        if not targets:
            logger.warning("No company targets resolved. "
                           "Check server allowlist and/or user's access profile.")
            return {"results": []}

        # Build base filter
        base_filter = {"resource_type": {"$eq": resource_type}} if resource_type else None

        # Query each namespace with per-company job restrictions
        all_matches: List[Dict[str, Any]] = []
        for cid, ns in targets:
            pine_filter = _company_filter_for(cid, base_filter)
            if pine_filter is None:
                # No access to this company (or no jobs)
                continue
            try:
                res = await _search_one_namespace(ns, qvec, top_k, pine_filter)
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
            snippet = meta.get("snippet") or ""  # stored short readable text at ingestion
            url = meta.get("body_url") or ""     # if you store URLs
            out.append({
                "id": m.get("id"),
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
        returns the first match found (that the user is allowed to see).
        """
        if not id:
            raise ValueError("id is required")

        targets = _resolve_company_targets(company_id, company_name)

        if not targets:
            # Fallback: legacy single namespace if configured AND (no server/user allowlist flow)
            if PINECONE_NAMESPACE_FALLBACK and not ALLOWED_COMPANIES and not ACCESS_PROFILE.companies:
                targets = [(-1, PINECONE_NAMESPACE_FALLBACK)]
            else:
                raise ValueError("No company targets resolved for fetch().")

        # Try each namespace until a permitted match is found
        for _, ns in targets:
            fetched = await asyncio.to_thread(index.fetch, ids=[id], namespace=ns)
            vectors = fetched.get("vectors") or {}
            item = vectors.get(id)
            if not item:
                continue
            meta = item.get("metadata", {}) or {}

            # Post-fetch permission guard
            if not _is_allowed(meta.get("company_id"), meta.get("company_jobposting_id")):
                continue  # found in this ns but the user can't see this job; try others

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

        raise ValueError(f"No permitted item found for id={id} in allowed namespaces.")

    return mcp

# ---------- Main ----------

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required (for query embeddings)")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required")

    logger.info(f"Using Pinecone index '{PINECONE_INDEX}'")

    # Optional sanity check: dimension matches embed model
    desc = pc.describe_index(PINECONE_INDEX)
    idx_dim = getattr(desc, "dimension", None) or desc.get("dimension")
    if idx_dim != INDEX_DIM:
        raise ValueError(
            f"Index dimension {idx_dim} != INDEX_DIM {INDEX_DIM}. "
            f"Update INDEX_DIM or recreate the index with the right size."
        )

    # Log server allowlist and user profile summary
    if ALLOWED_COMPANIES:
        logger.info(f"Server allowlist: {ALLOWED_COMPANIES}")
    elif PINECONE_NAMESPACE_FALLBACK:
        logger.info(f"Legacy single-namespace mode: '{PINECONE_NAMESPACE_FALLBACK}'")
    else:
        logger.info("No server allowlist set (PINECONE_ALLOWED_COMPANIES unset). Using only user access profile.")

    logger.info(
        "User access profile -> user_id=%s, companies=%s",
        ACCESS_PROFILE.user_id,
        {cid: {"role": sc.role, "all_jobs": sc.all_jobs, "jobs": (sorted(sc.job_ids) if sc.job_ids else None)}
         for cid, sc in ACCESS_PROFILE.companies.items()}
    )

    server = create_server()
    server.run(transport="http", port=8000)

if __name__ == "__main__":
    main()