import json
import re
import os
import logging
import requests
import sqlparse

from custom_slm import slm
from mahadbt_qdrant_schema_retriever import SmartSchemaRetrieverQdrant
from config import config_path
from config.logging_config import setup_logging
from helper_functions.mahadbt_helper_chat_history_test import execute_query

setup_logging()
logger = logging.getLogger("qa_api")

# ─────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────
CHAT_HISTORY = []
MAX_HISTORY = 3


def update_chat_history(user: str, assistant: str):
    CHAT_HISTORY.append({"user": user, "assistant": assistant})
    if len(CHAT_HISTORY) > MAX_HISTORY:
        CHAT_HISTORY.pop(0)


def get_chat_context() -> str:
    return "\n".join(
        f"User: {t['user']}\nAssistant: {t['assistant']}"
        for t in CHAT_HISTORY
    ).strip()


def extract_user_turns_only(chat_context: str) -> str:
    """Return only the user lines from a chat context string."""
    return "\n".join(
        line.replace("User:", "").strip()
        for line in chat_context.splitlines()
        if line.strip().startswith("User:")
    )


# ─────────────────────────────────────────
# CONFIG & SCHEMA RETRIEVER
# ─────────────────────────────────────────
with open(os.path.join(config_path, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

schema_retriever = SmartSchemaRetrieverQdrant(
    qdrant_url="http://localhost:6333",
    embed_model="all-MiniLM-L6-v2",
)

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def make_json_safe(obj):
    """Recursively convert numpy/pandas scalars to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    return obj


def fix_table_names(query: str) -> str:
    """Ensure every table reference is schema-qualified."""
    SCHEMA = "dbt_dashboard"
    TABLES = ["tbldashboard_2024-2025"]
    for t in TABLES:
        qualified = f'{SCHEMA}."{t}"'
        if qualified not in query:
            query = re.sub(
                rf'\b"?{re.escape(t)}"?\b',
                qualified,
                query,
            )
    return query


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ─────────────────────────────────────────
# SQL VALIDATOR
# ─────────────────────────────────────────
FORBIDDEN_KEYWORDS = {"drop", "delete", "insert", "update", "alter"}

# ─────────────────────────────────────────
# SCHEMA / SEMANTIC CONTEXT
# ─────────────────────────────────────────
ID_COLUMNS_BLACKLIST = {
    "districtid", "schemeid", "talukaid", "departmentid",
    "instituteid", "stateid", "applicantdistrictid", "applicanttalukaid",
    "divisionid", "categoryid", "religionid", "casteid",
    "totalreceived", "applicanttaluka", "applicantdistrict",
}

LABEL_KEYWORD_MAP = {
    "department": ["departmentname"],
    "scheme":     ["schemename"],
    "institute":  ["institutename"],
    "district":   ["districtname"],
    "state":      ["statename"],
    "taluk":      ["talukaname"],
    "division":   ["divisionname"],
}

VALUE_HINTS = {
    "financialyear": ["F.Y.2022-2023", "F.Y.2023-2024", "F.Y.2024-2025"],
}


def _gather_semantic_context(query: str, top_k_columns: int = 15) -> str:
    q_vec     = schema_retriever.encode_query(query)
    ctx_parts = []

    # --- Relevant columns ---
    try:
        column_hits = schema_retriever.search_columns(q_vec, limit=top_k_columns)
        seen, selected = set(), []

        for h in column_hits:
            cname = h.payload["column_name"]
            if cname in ID_COLUMNS_BLACKLIST or cname in seen:
                continue
            seen.add(cname)
            selected.append(cname)
            if len(selected) >= top_k_columns:
                break

        # Inject label columns based on keywords in the question
        query_lower = query.lower()
        for keyword, inject_cols in LABEL_KEYWORD_MAP.items():
            if keyword in query_lower:
                for col in inject_cols:
                    if col not in seen:
                        seen.add(col)
                        selected.insert(0, col)

        if selected:
            ctx_parts.append("### Relevant Columns")
            ctx_parts.extend(f"- {c}" for c in selected)

    except Exception as e:
        logger.warning("[Semantic Context - Columns] %s", e)

    # --- Metadata snippet ---
    try:
        meta_hits = schema_retriever.search_metadata(q_vec, limit=1)
        if meta_hits:
            meta = meta_hits[0].payload["metadata"]
            if isinstance(meta, list) and meta:
                table_info = meta[0]
                filtered_columns = [
                    {"column_name": col["column_name"], "description": col.get("description")}
                    for col in table_info.get("columns", [])[:15]
                    if col.get("column_name") not in ID_COLUMNS_BLACKLIST
                ]
                ctx_parts.append("\n### Metadata Snippet")
                ctx_parts.append(
                    json.dumps({"table": table_info.get("table_name"), "columns": filtered_columns}, indent=2)
                )
    except Exception as e:
        logger.warning("[Semantic Context - Metadata] %s", e)

    # --- Value hints ---
    ctx_parts.append("\n### Sample Values")
    for col, values in VALUE_HINTS.items():
        ctx_parts.append(f"{col}:")
        ctx_parts.extend(f"- {v}" for v in values)

    return "\n".join(ctx_parts)


def _get_optimized_schema(query: str) -> str:
    try:
        schema = schema_retriever.get_minimal_schema_for_query(query)
        return f'TABLE: tbldashboard_2024-2025\n\nCOLUMNS:\n{schema}'.strip()
    except Exception as e:
        logger.warning("[Schema Optimization] %s", e)
        return ""


# ─────────────────────────────────────────
# QUESTION REWRITER
# ─────────────────────────────────────────
def rewrite_followup_question(user_question: str, chat_context: str) -> str:
    if not chat_context:
        return user_question

    clean_context = extract_user_turns_only(chat_context)

    system_instruction = (
        "You are a question rewriting assistant. Your ONLY job is to rewrite or return questions as plain English text.\n\n"
        "Given a chat history (user questions only) and a new user question:\n"
        "- If the new question is a follow-up or refers to something in the chat history → "
        "rewrite it as a fully self-contained English question.\n"
        "- If the new question is completely independent → return it exactly as-is.\n\n"
        "STRICT RULES:\n"
        "- Output ONLY the rewritten question as plain text.\n"
        "- Do NOT output SQL, JSON, code, or any structured format.\n"
        "- Do NOT answer the question.\n"
        "- Do NOT include quotes, labels, or prefixes like 'Rewritten:' or 'Answer:'."
    )

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Previous user questions:\n{clean_context}\n\n"
        f"New question: {user_question}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    rewritten = slm.invoke(prompt)["choices"][0]["text"].strip()

    # Safety fallback: if the model returns SQL/JSON, use the original
    if rewritten.startswith("{") or rewritten.upper().startswith("SELECT"):
        return user_question

    return rewritten


# ─────────────────────────────────────────
# SQL GENERATION
# ─────────────────────────────────────────
def _build_sql_system_prompt(optimized_schema: str, semantic_context: str, question: str) -> str:
    return f"""
ROLE:
You are a SQL Generation Agent for the Maharashtra DBT Scholarship System.

TASK:
Generate an accurate PostgreSQL query that answers the user's question using ONLY
the provided SCHEMA and SEMANTIC_CONTEXT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN SELECTION  [STRICT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every column in SEMANTIC_CONTEXT:
  1. Find its description in SCHEMA.
  2. Ask: "Does this description directly match what the user is asking for?"
     → YES → include it.  NO → discard it.
Being present in SEMANTIC_CONTEXT alone is NOT sufficient to include a column.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE NAMING  [CRITICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always schema-qualify every table reference.
- Wrap table names with hyphens in double quotes.
  Example: dbt_dashboard."tbldashboard_2024-2025"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGGREGATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use SUM() / COUNT(*) for aggregations; always pair with GROUP BY.
- "how many distinct / number of unique" → COUNT(DISTINCT <col>), never SELECT DISTINCT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANKING  (top / best / highest / lowest / most)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Highest / Most  → ORDER BY <col> DESC
- Lowest / Least  → ORDER BY <col> ASC
- LIMIT 1 only for a single top/bottom result.
- LIMIT N only when the user specifies a number (e.g. "top 5 districts").
- No LIMIT for "district-wise" / "for each" queries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILTER / WHERE  [STRICT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Add a WHERE clause ONLY if the user explicitly names a filter value.
  → district name → filter by districtname
  → taluka name   → filter by talukaname
  → scheme name   → filter by schemename
  → division      → filter by divisionname
  → department    → filter by departmentname
Do NOT invent or assume filter values.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALUE NORMALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- financialyear is stored as 'F.Y.YYYY-YYYY'.
- Convert user input like '2024-25' → 'F.Y.2024-2025'.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSTGRESQL CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Target: PostgreSQL 13.
- Use double quotes for identifiers, NOT backticks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT  [STRICT — NON-NEGOTIABLE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON:
{{
  "query": "SELECT ..."
}}
No raw SQL. No ```sql blocks. No explanations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA:
{optimized_schema}

SEMANTIC_CONTEXT:
{semantic_context}

USER QUESTION:
{question}
""".strip()


def execute_db_pipeline(user_question: str) -> dict:
    """Rewrite question → build schema/context → generate SQL → return parsed dict."""
    chat_context      = get_chat_context()
    rewritten         = rewrite_followup_question(user_question, chat_context)
    optimized_schema  = _get_optimized_schema(rewritten)
    semantic_context  = _gather_semantic_context(rewritten)

    logger.info("[Rewritten] %s", rewritten)
    logger.info("[Schema]\n%s", optimized_schema)
    logger.info("[Semantic context]\n%s", semantic_context)

    system_instruction = _build_sql_system_prompt(optimized_schema, semantic_context, rewritten)

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
        f"<|im_start|>user\n{rewritten}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    raw = strip_think_tags(slm.invoke(prompt)["choices"][0]["text"])
    logger.debug("[Raw model output]\n%s", raw)

    # --- Primary: parse JSON ---
    parsed = extract_json(raw)
    if parsed and "query" in parsed:
        parsed["query"] = fix_table_names(parsed["query"])
        return parsed

    # --- Fallback: extract bare SQL ---
    sql_match = re.search(r"SELECT.*?;", raw, re.DOTALL | re.IGNORECASE)
    if sql_match:
        logger.warning("[DB Pipeline] Recovered SQL without JSON wrapper.")
        return {"query": fix_table_names(sql_match.group().strip())}

    raise ValueError(f"Model did not return valid JSON or SQL.\nRaw output:\n{raw}")


# ─────────────────────────────────────────
# NATURAL LANGUAGE RESPONSE
# ─────────────────────────────────────────
def natural_language_response(user_question: str, db_payload) -> str:
    safe_payload = make_json_safe(
        db_payload.to_dict(orient="records") if hasattr(db_payload, "to_dict") else db_payload
    )

    system_instruction = (
        "ROLE: You are a database result verbalization engine.\n\n"
        "TASK: Generate ONE correct English sentence that directly answers the user's question "
        "using only the provided DB data.\n\n"
        "RULES:\n"
        "- Use ONLY the provided DB data.\n"
        "- Do NOT invent values.\n"
        "- If multiple rows exist, summarize appropriately.\n"
        "- If no rows exist, say the data is unavailable.\n\n"
        "OUTPUT: Exactly ONE sentence. Starts with a capital letter. Ends with a full stop.\n\n"
        f"User Question:\n{user_question}\n\n"
        f"DB Response:\n{json.dumps(safe_payload, indent=2)}"
    )

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\nFinal Answer: "
    )

    answer = slm.invoke(prompt)["choices"][0]["text"].strip()
    update_chat_history(user_question, answer)
    return answer


# ─────────────────────────────────────────
# INTELLIGENCE / SMALL-TALK PIPELINE
# ─────────────────────────────────────────
def execute_intelligence_pipeline(user_question: str) -> str:
    system_instruction = (
        "ROLE: You are a Government Assistant.\n\n"
        "TASK: Analyze the user message and respond appropriately.\n\n"
        "RULE 1 — GREETINGS:\n"
        "If the message is a greeting (e.g. 'hello', 'hey', 'how are you'), "
        "respond politely and ask how you can help (1–2 lines max).\n\n"
        "RULE 2 — ABUSE:\n"
        "Respond EXACTLY: Please be respectful.\n\n"
        "RULE 3 — ANYTHING ELSE:\n"
        "Respond EXACTLY: I can help you with insights, KPI reports, "
        "and analytics from the existing DBT database.\n\n"
        f"User:\n{user_question}"
    )

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    answer = slm.invoke(prompt)["choices"][0]["text"].strip()
    update_chat_history(user_question, answer)
    return answer


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    question = "how many schemes are there in the financial year 2024-25?"

    db_result = execute_db_pipeline(question)
    sql = db_result.get("query", "")
    logger.info("[Generated SQL] %s", sql)

    rows, error = execute_query(sql)
    if error:
        logger.error("[Query Error] %s", error)
    else:
        logger.info("[Rows] %s", rows)
        print(natural_language_response(question, rows))
