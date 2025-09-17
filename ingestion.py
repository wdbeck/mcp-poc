import os
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# ------------------ DB CONFIG ------------------

PG = dict(
    host='localhost',
    port=5433,
    dbname='poc_mcp_data',
    user='postgres',
    password='april0415',
)

# ------------------ INGEST CONFIG (TEST VALUES) ------------------

INDEX_NAME   = 'poc-discovered-data-query'
COMPANY_ID   = 113
JOB_ID       = 3414
CANDIDATE_ID = 41671

# ------------------ CLIENTS ------------------

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  # same project/key as your MCP server
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])   # text-embedding-3-large (3072 dim)

# Create Pinecone index if missing (serverless; change region/cloud if desired)
DIM, METRIC = 3072, "cosine"
existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME, dimension=DIM, metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # wait until ready
    while True:
        if pc.describe_index(INDEX_NAME).status["ready"]:
            break
        time.sleep(1)

index = pc.Index(INDEX_NAME)

# ----------------------------------------------------------------------
# HARD-CODED LABEL DICTIONARIES (candidate activity only; job activity removed)
# ----------------------------------------------------------------------

CANDIDATE_ACTIVITY_LABELS = {
    "added_to_favorites": "Added to favorites",
    "application_completed": "Application completed",
    "application_requested": "Application requested",
    "assessment_assigned_manually": "Assessment assigned manually",
    "assessment_assigned_via_assessment_link": "Assessment assigned via assessment link",
    "assessment_assigned_via_assessment_status": "Assessment assigned via status change",
    "assessment_unassigned_manually": "Assessment unassigned from candidate",
    "candidate_added_manually": "Candidate added to the job manually",
    "candidate_added_via_application_form": "Candidate added to the job via application form",
    "candidate_added_via_assessment_link": "Candidate added to the job via assessment link",
    "candidate_added_via_email": "Candidate added to the job via email-to-apply",
    "candidate_added_via_job_board": "Candidate added to the job via job board",
    "candidate_copied": "Candidate copied to another job",
    "candidate_copied_from": "Candidate copied to the job from another job",
    "candidate_copied_to": "Candidate copied to another job",
    "candidate_document_added": "Attachment added to candidate profile",
    "candidate_document_deleted": "Attachment deleted from candidate profile",
    "candidate_moved": "Candidate moved to another job",
    "candidate_moved_to_status": "Candidate's status changed",
    "candidate_note_added": "Candidate note added",
    "candidate_note_deleted": "Candidate note deleted",
    "candidate_note_edited": "Candidate note edited",
    "candidate_scorecard_note_added": "Scorecard note added",
    "candidate_scorecard_note_deleted": "Scorecard note deleted",
    "candidate_scorecard_note_edited": "Scorecard note edited",
    "candidate_scorecard_rating_added": "Scorecard rating added",
    "candidate_scorecard_rating_edited": "Scorecard rating edited",
    "candidate_screener_added": "Candidate screener questions added",
    "candidate_sequence_failed": "Candidate sequence failed",
    "candidate_submitted_reference_contact": "Reference contact submitted",
    "email_sending_to_candidate": "Email sent to candidate",
    "given_star_rating": "Candidate given a star rating",
    "interview_questions_sent": "Interview questions sent",
    "interview_schedule_added": "Interview scheduled",
    "interview_schedule_completed": "Scheduled interview completed",
    "interview_schedule_edited": "Scheduled interview edited",
    "one_way_interview_completed": "One-way interview completed",
    "reference_completed": "Reference completed questions",
    "reference_questions_sending": "Reference questions sent to reference",
    "reference_questions_stop_sending": "Stopped sending reference questions to reference",
    "references_requested_manually": "References requested manually",
    "references_requested_via_status": "References requested via status change",
    "removed_from_favorites": "Removed from favorites",
    "resume_requested": "Resume requested",
    "resume_updated": "Resume updated",
    "screening_questions_completed": "Screening questions completed",
    "sms_sending_to_candidate": "Sent text message to candidate"
}

NOTE_EVENT_TYPES = {"candidate_note_added", "candidate_note_edited", "candidate_note_deleted"}

# ------------------ MISC HELPERS ------------------

def truncate(text: str, max_chars: int = 20000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def clean_meta(d: dict) -> dict:
    """
    Recursively clean metadata to JSON-friendly primitives.
    Pinecone metadata must be string, number, boolean, or **list of strings**.
    This function preserves only those types; nested objects are stringified by caller if needed.
    """
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, list):
            # keep only string items
            out = []
            for x in v:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, (int, float, bool)):
                    out.append(str(x))
                else:
                    out.append(str(x))
            return out
        if isinstance(v, dict):
            # Stringify dicts at this layer (caller should usually pre-stringify if needed)
            return json.dumps(v, ensure_ascii=False)
        # fallback to string
        return str(v)

    out = {}
    for k, v in d.items():
        vv = _clean(v)
        if vv is not None:
            out[str(k)] = vv
    return out


def embed(text: str) -> List[float]:
    r = openai.embeddings.create(model="text-embedding-3-large", input=text or "")
    return r.data[0].embedding


def format_actor(name: str | None, email: str | None) -> str:
    name = (name or "").strip()
    email = (email or "").strip()
    if name and email:
        return f"{name} ({email})"
    if name:
        return name
    if email:
        return email
    return ""


def namespace_for_company(company_id: int) -> str:
    return f"company_{int(company_id)}"

# ------------------ FETCHERS ------------------


def fetch_one_job(conn, job_id: int):
    """Join to company to get company_name in addition to company_id."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT 
                jp.company_id,
                c.company_name,
                jp.company_jobposting_id,
                jp.jobpost_title,
                jp.jobpost_description,
                jp.advertisement,
                jp.specification,
                jp.jobpost_city,
                jp.jobpost_state
            FROM public.company_jobposting jp
            JOIN public.company c ON c.company_id = jp.company_id
            WHERE jp.company_jobposting_id = %s
            """,
            (job_id,),
        )
        return cur.fetchone()


def fetch_one_candidate(conn, candidate_id: int):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT 
                cjc.company_jobposting_candidate_id,
                cjc.company_jobposting_id,
                cjc.first_name, 
                cjc.last_name,
                cjc.parsed_document,       -- JSON blob from resume parser; we will pull detailResume
                cjc.primary_source
            FROM public.company_jobposting_candidate cjc
            WHERE cjc.company_jobposting_candidate_id = %s
            """,
            (candidate_id,),
        )
        row = cur.fetchone()
        if not row:
            return None

        # Extract resume text from parsed_document.detailResume if available
        resume_text = ""
        pd = row.get("parsed_document")
        if pd:
            try:
                if isinstance(pd, str):
                    pd_json = json.loads(pd)
                else:
                    pd_json = pd
                resume_text = pd_json.get("detailResume") or ""
            except Exception:
                # fallback: if parsed_document is already plain text
                if isinstance(pd, str):
                    resume_text = pd
        row["detail_resume"] = resume_text
        return row


def fetch_candidate_activity(conn, candidate_id: int):
    """Candidate activity events (humanized + actor)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
              ca.company_jobposting_candidate_activity_id AS activity_id,
              ca.activity_type,
              ca.company_jobposting_status_name,
              ca.moved_from, 
              ca.moved_to,
              ca.date_added,
              ca.manager_id AS user_id,
              CONCAT_WS(' ', u.user_first_name, u.user_last_name) AS user_full_name,
              u.user_email_address
            FROM public.company_jobposting_candidate_activity ca
            LEFT JOIN public."user" u ON u.user_id = ca.manager_id
            WHERE ca.company_jobposting_candidate_id = %s
            ORDER BY ca.date_added DESC
            """,
            (candidate_id,),
        )
        return cur.fetchall()


def fetch_candidate_skills(conn, candidate_id: int) -> List[str]:
    """Skills from company_jobposting_candidate_skill."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT skill
            FROM public.company_jobposting_candidate_skill
            WHERE company_jobposting_candidate_id = %s
            ORDER BY skill ASC
            """,
            (candidate_id,),
        )
        return [r["skill"] for r in cur.fetchall()]


def fetch_candidate_assessments(conn, candidate_id: int) -> List[Dict[str, Any]]:
    """Assessments with names, status, dates, scores."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT 
              ca.company_jobposting_candidate_assessment_id AS id,
              ca.assessment_id,
              a.assessment_name,
              ca.date_assigned,
              ca.assessment_complete,
              ca.completion_date,
              ca.assessment_score
            FROM public.company_jobposting_candidate_assessment ca
            JOIN public.assessment a ON a.assessment_id = ca.assessment_id
            WHERE ca.company_jobposting_candidate_id = %s AND ca.active = true
            ORDER BY ca.date_assigned DESC
            """,
            (candidate_id,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            status = "completed" if (r.get("assessment_complete") or 0) else "assigned"
            out.append(
                {
                    "id": int(r["id"]),
                    "assessment_id": int(r["assessment_id"]),
                    "name": r.get("assessment_name") or "",
                    "status": status,
                    "date_assigned": r.get("date_assigned").isoformat() if r.get("date_assigned") else None,
                    "completion_date": r.get("completion_date").isoformat() if r.get("completion_date") else None,
                    "score": r.get("assessment_score"),
                }
            )
        return out


def fetch_candidate_notes(conn, candidate_id: int) -> List[Dict[str, Any]]:
    """Notes left on the candidate; used to enrich note-related activity events with note text."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT 
              n.company_jobposting_candidate_note_id AS note_id,
              n.note,
              n.date_added,
              n.user_id,
              CONCAT_WS(' ', u.user_first_name, u.user_last_name) AS user_full_name,
              u.user_email_address
            FROM public.company_jobposting_candidate_note n
            LEFT JOIN public."user" u ON u.user_id = n.user_id
            WHERE n.company_jobposting_candidate_id = %s
            ORDER BY n.date_added DESC
            """,
            (candidate_id,),
        )
        return cur.fetchall()

# ------------------ RENDERERS ------------------


def humanize_activity_type(code: str) -> str:
    if not code:
        return "Activity"
    return CANDIDATE_ACTIVITY_LABELS.get(code.strip(), code.replace("_", " ").capitalize())


def render_job_text(job: dict) -> str:
    parts = [
        job.get("jobpost_title") or "",
        job.get("jobpost_description") or "",
        job.get("advertisement") or "",
        job.get("specification") or "",
        f"{job.get('jobpost_city') or ''} {job.get('jobpost_state') or ''}".strip(),
    ]
    return " ".join(p for p in parts if p)


def render_assessments_summary(assessments: List[Dict[str, Any]]) -> str:
    if not assessments:
        return ""
    lines = []
    for a in assessments:
        bits = [f"{a.get('name','Assessment')}"]
        if a.get("status") == "completed":
            when = a.get("completion_date") or a.get("date_assigned")
            if when:
                bits.append(f"completed {when[:10]}")
        else:
            when = a.get("date_assigned")
            if when:
                bits.append(f"assigned {when[:10]}")
        if a.get("score") is not None:
            bits.append(f"score {a['score']}")
        lines.append(" (" + ", ".join(bits) + ")")
    return "Assessments:" + "".join(lines)


def assessments_to_strings(assessments: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for a in assessments:
        parts = [a.get("name") or "Assessment"]
        status = a.get("status")
        if status:
            parts.append(status)
        if status == "completed" and a.get("completion_date"):
            parts.append(f"completed {a['completion_date'][:10]}")
        elif a.get("date_assigned"):
            parts.append(f"assigned {a['date_assigned'][:10]}")
        if a.get("score") is not None:
            parts.append(f"score {a['score']}")
        out.append(", ".join(parts))
    return out


def render_candidate_text(c: dict, skills: List[str], assessments: List[Dict[str, Any]], resume_text: str) -> str:
    name = " ".join(x for x in [c.get("first_name"), c.get("last_name")] if x)
    parts = [
        f"Candidate: {name}",
        ("Skills: " + ", ".join(skills)) if skills else "",
        render_assessments_summary(assessments),
        truncate(resume_text or ""),  # include resume text from parsed_document.detailResume
        f"Source: {c.get('primary_source')}" if c.get("primary_source") else "",
    ]
    return " ".join(p for p in parts if p)


def render_candidate_activity_event(ev: dict) -> str:
    ts = ev["date_added"].strftime("%Y-%m-%d %H:%M")
    label = humanize_activity_type(ev.get("activity_type"))
    moved = ""
    if ev.get("moved_from") or ev.get("moved_to"):
        moved = f" ({ev.get('moved_from') or ''} → {ev.get('moved_to') or ''})"
    status = f" — {ev['company_jobposting_status_name']}" if ev.get("company_jobposting_status_name") else ""
    actor = format_actor(ev.get("user_full_name"), ev.get("user_email_address"))
    actor_suffix = f" — {actor}" if actor else ""
    note_part = ""
    if ev.get("note_text"):
        note_text = (ev.get("note_text") or "").strip()
        if note_text:
            note_part = f" — Note: {note_text}"
    return f"[{ts}] {label}{status}{moved}{actor_suffix}{note_part}"

# ------------------ ENRICHMENT ------------------


def enrich_notes_into_activity(cand_act_rows: List[Dict[str, Any]], note_rows: List[Dict[str, Any]]) -> None:
    """
    Attach note text from company_jobposting_candidate_note to the corresponding
    candidate activity rows WHEN the activity_type is one of NOTE_EVENT_TYPES.

    Matching strategy (heuristic):
      1) Exact match on (user_id, date_added to the second).
      2) If no exact, choose closest note within ±5 seconds for the same user.
      3) If still none, attach the most recent earlier note by the same user.
    """
    if not cand_act_rows or not note_rows:
        return

    # Normalize times to seconds resolution
    def norm(ts: datetime) -> datetime:
        if not isinstance(ts, datetime):
            return ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.replace(microsecond=0)

    notes = [dict(n, date_added=norm(n.get("date_added"))) for n in note_rows]

    # Build index by user
    by_user: Dict[Optional[int], List[Dict[str, Any]]] = {}
    for n in notes:
        by_user.setdefault(n.get("user_id"), []).append(n)
    for lst in by_user.values():
        lst.sort(key=lambda x: x.get("date_added") or datetime.min, reverse=True)

    for ev in cand_act_rows:
        if (ev.get("activity_type") or "").strip() not in NOTE_EVENT_TYPES:
            continue
        u = ev.get("user_id")
        ev_ts = norm(ev.get("date_added"))
        candidates = by_user.get(u) or []

        # 1) exact timestamp match
        match = next((n for n in candidates if n.get("date_added") == ev_ts), None)

        # 2) nearest within ±5s
        if not match and ev_ts:
            nearest = None
            nearest_dt = None
            for n in candidates:
                nd = n.get("date_added")
                if not nd:
                    continue
                delta = abs((ev_ts - nd).total_seconds())
                if delta <= 5 and (nearest_dt is None or delta < nearest_dt):
                    nearest = n
                    nearest_dt = delta
            match = nearest

        # 3) most recent earlier
        if not match and ev_ts:
            earlier = [n for n in candidates if n.get("date_added") and n.get("date_added") <= ev_ts]
            if earlier:
                earlier.sort(key=lambda x: x.get("date_added"), reverse=True)
                match = earlier[0]

        if match:
            ev["note_text"] = match.get("note")
            # Also carry who (already in ev) but ensure from note if missing
            if not ev.get("user_full_name") and match.get("user_full_name"):
                ev["user_full_name"] = match.get("user_full_name")
            if not ev.get("user_email_address") and match.get("user_email_address"):
                ev["user_email_address"] = match.get("user_email_address")

# ------------------ UPSERT (per-event docs; per-company namespace) ------------------


def upsert_docs(job: dict,
                cand: dict,
                cand_act_rows: List[Dict[str, Any]],
                note_rows: List[Dict[str, Any]],
                skills: List[str],
                assessments: List[Dict[str, Any]],
                resume_text: str) -> None:
    """
    Creates:
      - 1 vector: job
      - 1 vector: candidate (now includes skills, assessments, and resume text)
      - N vectors: candidate_activity events (one per row) — if an event is a note, its text is embedded and stored here
    Each event vector carries:
      - parent_vec_id -> "cand:{candidate_id}"
      - job/candidate IDs (for joins/filters)
    """
    namespace = namespace_for_company(job["company_id"])  # hard tenant isolation

    # --- parent docs
    job_vec_id  = f"job:{job['company_jobposting_id']}"
    cand_vec_id = f"cand:{cand['company_jobposting_candidate_id']}"

    job_text  = render_job_text(job)
    cand_text = render_candidate_text(cand, skills, assessments, resume_text)

    # Normalize primary_source a bit for later filtering (optional)
    src = (cand.get("primary_source") or "").strip().lower()
    ALIASES = {"indeed.com": "indeed", "indeed apply": "indeed"}
    source_norm = ALIASES.get(src, src)

    company_name = job.get("company_name")

    # Prepare Pinecone-compatible metadata for assessments
    assessments_list_for_meta = assessments_to_strings(assessments)
    assessments_json_for_meta = json.dumps(assessments, ensure_ascii=False) if assessments else None

    parent_vectors = [
        {
            "id": job_vec_id,
            "values": embed(job_text),
            "metadata": clean_meta({
                "resource_type": "job",
                "company_id": int(job["company_id"]),
                "company_name": company_name,
                "company_jobposting_id": int(job["company_jobposting_id"]),
                "title": job.get("jobpost_title") or "",
            }),
        },
        {
            "id": cand_vec_id,
            "values": embed(cand_text),
            "metadata": clean_meta({
                "resource_type": "candidate",
                "company_id": int(job["company_id"]),
                "company_name": company_name,
                "company_jobposting_id": int(job["company_jobposting_id"]),
                "company_jobposting_candidate_id": int(cand["company_jobposting_candidate_id"]),
                "primary_source": source_norm,
                "title": " ".join(x for x in [cand.get("first_name"), cand.get("last_name")] if x),
                # Enrichments (Pinecone-safe)
                "skills": skills or [],                               # list[str]
                "assessments": assessments_list_for_meta or [],        # list[str]
                "assessments_json": assessments_json_for_meta,         # stringified JSON (optional)
                "detail_resume": truncate(resume_text or ""),
            }),
        },
    ]

    # --- enrich activity rows with notes where applicable
    enrich_notes_into_activity(cand_act_rows, note_rows)

    # --- candidate activity: one vector per event (note text included when present)
    cand_event_vectors: List[Dict[str, Any]] = []
    for ev in cand_act_rows:
        event_text = render_candidate_activity_event(ev)
        event_id = ev.get("activity_id") or ev["date_added"].strftime("%Y%m%d%H%M%S")
        vec_id = f"cand_act:{cand['company_jobposting_candidate_id']}:{event_id}"
        cand_event_vectors.append({
            "id": vec_id,
            "values": embed(event_text),
            "metadata": clean_meta({
                "resource_type": "candidate_activity_event",
                "parent_vec_id": cand_vec_id,
                "company_id": int(job["company_id"]),
                "company_name": company_name,
                "company_jobposting_id": int(job["company_jobposting_id"]),
                "company_jobposting_candidate_id": int(cand["company_jobposting_candidate_id"]),
                "event_id": str(event_id),
                "date_added": ev["date_added"].isoformat() if ev.get("date_added") else None,
                "activity_type_raw": ev.get("activity_type"),
                "activity_type": humanize_activity_type(ev.get("activity_type")),
                "status_name": ev.get("company_jobposting_status_name"),
                "moved_from": ev.get("moved_from"),
                "moved_to": ev.get("moved_to"),
                # actor fields
                "user_id": ev.get("user_id"),
                "user_full_name": ev.get("user_full_name"),
                "user_email_address": ev.get("user_email_address"),
                # note enrichment
                "note_text": ev.get("note_text"),
                "title": " ".join(x for x in [cand.get("first_name"), cand.get("last_name")] if x) + " — candidate activity",
                "snippet": event_text,
            }),
        })

    # Batch upsert
    to_upsert = parent_vectors + cand_event_vectors
    index.upsert(vectors=to_upsert, namespace=namespace)
    print(
        f"Upserted {len(to_upsert)} vectors to Pinecone namespace '{namespace}' "
        f"(company_name='{company_name}'). "
        f"Parents={len(parent_vectors)}, Candidate events={len(cand_event_vectors)}"
    )

# ------------------ MAIN ------------------


def main():
    with psycopg2.connect(**PG) as conn:
        job = fetch_one_job(conn, JOB_ID)
        if not job:
            raise SystemExit(f"No job found for ID {JOB_ID}")
        cand = fetch_one_candidate(conn, CANDIDATE_ID)
        if not cand:
            raise SystemExit(f"No candidate found for ID {CANDIDATE_ID}")
        if cand["company_jobposting_id"] != job["company_jobposting_id"]:
            print("Warning: candidate does not belong to the given job.")

        # Fetch enrichments
        skills = fetch_candidate_skills(conn, CANDIDATE_ID)
        assessments = fetch_candidate_assessments(conn, CANDIDATE_ID)
        notes = fetch_candidate_notes(conn, CANDIDATE_ID)
        cand_act_rows = fetch_candidate_activity(conn, CANDIDATE_ID)

    # Use parsed_document.detailResume (added to cand by fetch_one_candidate)
    resume_text = cand.get("detail_resume") or ""

    upsert_docs(job, cand, cand_act_rows, notes, skills, assessments, resume_text)


if __name__ == "__main__":
    main()