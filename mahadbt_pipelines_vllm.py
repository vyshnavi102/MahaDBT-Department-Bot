import json
import re
import os
import pandas as pd
import logging
import requests
import sqlparse
from mahadbt_qdrant_schema_retriever import SmartSchemaRetrieverQdrant
from config import config_path
from config.logging_config import setup_logging, log_email
from helper_functions.mahadbt_helper_chat_history_test import execute_query

setup_logging()

import logging

logger = logging.getLogger("qa_api")
# -------------------------------
# CHAT HISTORY
# -------------------------------
CHAT_HISTORY = []
MAX_HISTORY = 3

def update_chat_history(user, assistant):
    CHAT_HISTORY.append({"user": user, "assistant": assistant})
    if len(CHAT_HISTORY) > MAX_HISTORY:
        CHAT_HISTORY.pop(0)

def get_chat_context():
    context = ""
    for turn in CHAT_HISTORY:
        context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    return context.strip()

def rewrite_followup_question(user_question, chat_context):
    if not chat_context:
        return user_question

    # Strip assistant SQL responses from context — keep only user turns
    clean_context = extract_user_turns_only(chat_context)

    system_instruction = """
You are a question rewriting assistant. Your ONLY job is to rewrite or return questions as plain English text.

Given a chat history (user questions only) and a new user question:
- If the new question is a follow-up or refers to something in the chat history → rewrite it as a fully self-contained English question
- If the new question is completely independent → return it exactly as-is

STRICT RULES:
- Output ONLY the rewritten question or the original question as plain text
- Do NOT output SQL, JSON, code, or any structured format
- Do NOT answer the question
- Do NOT explain your reasoning
- Do NOT include quotes, labels, or prefixes like "Rewritten:" or "Answer:"
""".strip()

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Previous user questions:\n{clean_context}\n\n"
        f"New question: {user_question}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    out = model(prompt, max_tokens=100)
    rewritten = out["choices"][0]["text"].strip()

    # Safety fallback: if output looks like SQL/JSON, return original
    if rewritten.startswith("{") or rewritten.upper().startswith("SELECT"):
        return user_question

    return rewritten


def extract_user_turns_only(chat_context):
    """Extract only user messages from chat context, ignoring assistant SQL responses."""
    lines = chat_context.strip().split("\n")
    user_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("User:"):
            user_lines.append(line.replace("User:", "").strip())
    return "\n".join(user_lines)
# -------------------------------
# LOAD CONFIG
# -------------------------------
with open(os.path.join(config_path, "config.json"), "r", encoding="utf-8") as CONFIG:
    config = json.load(CONFIG)

# -------------------------------
# INITIALIZE SCHEMA RETRIEVER
# -------------------------------
schema_retriever = SmartSchemaRetrieverQdrant(
    qdrant_url="http://localhost:6333",
    embed_model="all-MiniLM-L6-v2"
)

# -------------------------------
# MODEL CONFIG
# -------------------------------
MODEL_PATH = "HuggingFaceTB/SmolLM3-3B"
MODEL_API = "http://localhost:8000"

TOP_P = 1.0
TOP_K = 1
TEMPERATURE = 0.0
STOP = ["<|im_end|>"]

# -------------------------------
# VLLM CLIENT
# -------------------------------
class VLLMClient:
    def __init__(self, base_url=MODEL_API, model_name=MODEL_PATH):
        self.url = f"{base_url}/v1/chat/completions"
        self.model = model_name

    def __call__(self, prompt, max_tokens=400):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "chat_template_kwargs": {"enable_thinking": False},
            "stop": STOP
        }

        response = requests.post(self.url, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        return {
            "choices": [
                {"text": data["choices"][0]["message"]["content"]}
            ]
        }

model = VLLMClient()


# -------------------------------
# SQL VALIDATOR
# -------------------------------
def validate_sql_query(query: str):
    if not query:
        return {"valid": False, "error": "Empty query"}

    q = query.lower()

    # safety
    if not q.startswith("select"):
        return {"valid": False, "error": "Only SELECT queries allowed"}

    forbidden = ["drop", "delete", "insert", "update", "alter"]
    if any(f in q for f in forbidden):
        return {"valid": False, "error": "Unsafe query detected"}

    try:
        parsed = sqlparse.parse(query)
        if not parsed:
            return {"valid": False, "error": "Invalid SQL syntax"}
    except Exception:
        return {"valid": False, "error": "SQL parsing failed"}

    return {"valid": True}

# -------------------------------
# SQL RETRY FIX
# -------------------------------
def regenerate_sql_with_feedback(user_question, error_msg, optimized_schema, semantic_context):

    system_instruction = f"""
ROLE:
You are a SQL correction agent.

TASK:
Fix the SQL query based on the error.

ERROR:
{error_msg}

RULES:
- Fix ONLY the SQL
- Use schema strictly
- Keep intent same

OUTPUT:
{{
  "query": "<CORRECTED_SQL>"
}}
""".strip()

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n\n"
        f"<DB_SCHEMA>\n{optimized_schema}\n</DB_SCHEMA>\n\n"
        f"<SEMANTIC_CONTEXT>\n{semantic_context}\n</SEMANTIC_CONTEXT>\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    out = model(prompt)
    raw = out["choices"][0]["text"]

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    return None


# Add this set at the top of _gather_semantic_context
ID_COLUMNS_BLACKLIST = {
    "districtid", "schemeid", "talukaid", "departmentid",
    "instituteid", "stateid", "applicantdistrictid", "applicanttalukaid",
    "divisionid", "categoryid", "religionid", "casteid","totalreceived","applicanttaluka","applicantdistrict"
}

def _gather_semantic_context(query: str, top_k_columns=15):
    q_vec = schema_retriever.encode_query(query)
    ctx_parts = []

    try:
        column_hits = schema_retriever.search_columns(q_vec, limit=top_k_columns)
        seen = set()
        selected = []

        for h in column_hits:
            cname = h.payload["column_name"]

            # ✅ Skip ID columns entirely
            if cname in ID_COLUMNS_BLACKLIST:
                continue

            if cname not in seen:
                seen.add(cname)
                selected.append(cname)
                if len(selected) >= top_k_columns:
                    break

        # ✅ Inject only NAME columns, no IDs
        LABEL_KEYWORD_MAP = {
            "department": ["departmentname"],
            "scheme":     ["schemename"],
            "institute":  ["institutename"],
            "district":   ["districtname"],
            "state":      ["statename"],
            "taluk":      ["talukaname"],
            "division":   ["divisionname"],
        }

        query_lower = query.lower()
        for keyword, inject_cols in LABEL_KEYWORD_MAP.items():
            if keyword in query_lower:
                for col in inject_cols:
                    if col not in seen:
                        seen.add(col)
                        selected.insert(0, col)

        if selected:
            ctx_parts.append("### Relevant Columns")
            for c in selected:
                ctx_parts.append(f"- {c}")

    except Exception as e:
        logger.warning("[Semantic Context - Columns Error] %s", e)

    try:
        meta_hits = schema_retriever.search_metadata(q_vec, limit=1)
        if meta_hits:
            meta = meta_hits[0].payload["metadata"]
            if isinstance(meta, list) and len(meta) > 0:
                table_info = meta[0]
                columns = table_info.get("columns", [])[:15]

                # ✅ Also filter ID columns from metadata snippet
                filtered_columns = [
                    {
                        "column_name": col.get("column_name"),
                        "description": col.get("description")
                    }
                    for col in columns
                    if col.get("column_name") not in ID_COLUMNS_BLACKLIST
                ]

                snippet = json.dumps(
                    {
                        "table": table_info.get("table_name"),
                        "columns": filtered_columns
                    },
                    indent=2
                )
                ctx_parts.append("\n### Metadata Snippet")
                ctx_parts.append(snippet)

    except Exception as e:
        logger.warning("[Semantic Context - Metadata Error] %s", e)

    # -------------------------------
    # ADD VALUE HINTS (IMPORTANT FIX)
    # -------------------------------
    VALUE_HINTS = {
        "financialyear": [
            "F.Y.2022-2023",
            "F.Y.2023-2024",
            "F.Y.2024-2025"
        ]
    }


    ctx_parts.append("\n### Sample Values")
    for col, values in VALUE_HINTS.items():
        ctx_parts.append(f"{col}:")
        for v in values:
            ctx_parts.append(f"- {v}")

    return "\n".join(ctx_parts)

def _run_schema_optimization(query: str):
    try:
        schema = schema_retriever.get_minimal_schema_for_query(query)
        return f"""
TABLE: tbldashboard_2024-2025

COLUMNS:
{schema}
""".strip()
    except Exception as e:
        logger.warning(f"[Schema Optimization Error] {e}")
        return ""


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj

def fix_table_names(query: str):
    TABLES = ["tbldashboard_2024-2025"]
    SCHEMA = "dbt_dashboard"

    for t in TABLES:
        # Skip if already schema-qualified
        if f'{SCHEMA}."{t}"' in query:
            continue

        # Replace only standalone table (quoted or unquoted)
        query = re.sub(
            rf'\b"?{re.escape(t)}"?\b',
            f'{SCHEMA}."{t}"',
            query
        )

    return query
# -------------------------------
# NATURAL LANGUAGE RESPONSE
# -------------------------------
def natural_language_response(user_question, db_payload):

    safe_payload = make_json_safe(db_payload)

    system_instruction = f"""
ROLE:
You are a database result verbalization engine.

TASK:
Analyze the database response and generate ONE correct English sentence
that directly answers the user's question.

RULES:
- Use ONLY the provided DB data
- Do NOT invent values
- Do NOT use placeholders
- If multiple rows exist, summarize appropriately
- If ranking is needed, infer it from rows
- If no rows exist, say data is unavailable

OUTPUT:
- Exactly ONE sentence
- Capital letter start
- Full stop end

User Question:
{user_question}

DB Response:
{json.dumps(safe_payload, indent=2)}
""".strip()

    prompt = f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n<|im_start|>assistant\nFinal Answer: "

    out = model(prompt, max_tokens=80)
    answer = out["choices"][0]["text"].strip()

    update_chat_history(user_question, answer)
    return answer

# -------------------------------
# VISUALIZATION PIPELINE
# -------------------------------
def execute_visualization_pipeline(user_question, response_db):

    system_instruction = """
ROLE:
You are a STRICT visualization payload generator for Government Scholarship Data.

TASK:
Analyze the data inside <RESPONSE_DB_FULL_DATA> and the user question.
Generate ONLY a JSON object describing the visualization AND a short insight summary.

------------------------------------------------
ABSOLUTE OUTPUT RULES (NON-NEGOTIABLE)
------------------------------------------------
- Output MUST be valid JSON
- Output MUST contain ONLY the JSON object
- NO markdown
- NO explanations
- NO comments
- NO extra text

------------------------------------------------
CHART TYPE SELECTION RULES (STRICT)
------------------------------------------------
- bar: comparisons
- line: trends over time
- pie: proportions
- scatter: correlation
- area: cumulative trends
- Use "table" IF:
   - The question mentions table

------------------------------------------------
REQUIRED JSON SCHEMA
------------------------------------------------
{
  "chart_type": "",
  "x": [],
  "y": [],
  "x_label": "",
  "y_label": "",
  "title": "",
  "insight": ""
}
""".strip()
    payload = response_db.to_dict(orient="records") if isinstance(response_db, pd.DataFrame) else response_db
    payload = make_json_safe(payload)

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n"
        f"<RESPONSE_DB_FULL_DATA>\n{json.dumps(payload, indent=2)}\n</RESPONSE_DB_FULL_DATA>\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return model(prompt, max_tokens=1200)["choices"][0]["text"].strip()
# -------------------------------
# DB PIPELINE (FINAL)
# -------------------------------
def execute_db_pipeline(user_question: str):
    logger = logging.getLogger("qa_api")
    chat_context = get_chat_context()
    print(f"Chat Context---", chat_context)
    logger.info(f"Chat context: {chat_context}")
    rewritten_question = rewrite_followup_question(user_question, chat_context)
    print(f"[Rewritten Question]: {rewritten_question}")
    logger.info(f"Rewritten Question: {rewritten_question}")
    optimized_schema = _run_schema_optimization(rewritten_question)
    print(f"optimized schema : {optimized_schema}")
    semantic_context = _gather_semantic_context(rewritten_question)
    print(f"semantic context : {semantic_context}")

    system_instruction = f"""
ROLE:
You are a SQL Generation Agent for the Maharashtra DBT Scholarship System.

TASK:
Generate an accurate PostgreSQL query that answers the user's question using ONLY
the provided SCHEMA and SEMANTIC_CONTEXT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COLUMN SELECTION PROCESS  [STRICT — NON-NEGOTIABLE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EVERY column in SEMANTIC_CONTEXT, do the following:
  1. Find that column's description in SCHEMA.
  2. Read the description.
  3. Ask: "Does this description directly match what the user is asking for?"
       → YES → include the column.
       → NO  → discard the column. Do not include it.
Being present in SEMANTIC_CONTEXT is NOT a reason to select a column.
ONLY the SCHEMA description decides.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE NAMING  [CRITICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Always include the schema name in every query.
- Wrap table names containing special characters (e.g. hyphens) in double quotes.
  Example: dbt_dashboard."tbldashboard_2024-2025"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGGREGATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use SUM() / COUNT(*) when aggregating.
- Always include GROUP BY when using aggregate functions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISTINCT COUNT RULES  [CRITICAL FIX]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If the question asks:
   "how many distinct <column>"
   "number of unique <column>"
   "count of unique <column>"

→ ALWAYS use:
   COUNT(DISTINCT <column>)

- NEVER return SELECT DISTINCT for count questions
- SELECT DISTINCT is ONLY for listing values, NOT counting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RANKING RULES  (TOP / BEST / HIGHEST / LOWEST / MOST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Highest / Most  → ORDER BY <col> DESC
- Lowest / Least  → ORDER BY <col> ASC
- Use LIMIT 1 only for a single top/bottom result.
- Use LIMIT N only when the user specifies a number (e.g. "top 5 districts").
- Do NOT add LIMIT for all-results queries (e.g. "district-wise", "for each").

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SORTING RULES  (ranked lists / sorted results)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Use ORDER BY to sort results.
- Do NOT add LIMIT unless explicitly requested.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILTER / WHERE CLAUSE RULES  [STRICT]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY add a WHERE condition if the user explicitly mentions a filter value
in their question.

  → If the user mentions a district   → filter by districtname
  → If the user mentions a taluka     → filter by talukaname
  → If the user mentions a scheme     → filter by schemename
  → If the user mentions a Division/division   → filter by divisionname
  → If the user mentions a Department/department  → filter by departmentname

DO NOT invent or assume any filter value that the user did not state.
DO NOT add schemename, divisionname, or any other filter
unless the user explicitly named it in the question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALUE NORMALIZATION RULES  [CRITICAL]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- financialyear values are stored as 'F.Y.YYYY-YYYY'
- If user gives '2024-25', convert it to 'F.Y.2024-2025'
- Always match database format exactly in WHERE clause

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSTGRESQL CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Target version: PostgreSQL 13.
- Do NOT use backticks — use double quotes for identifiers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT  [STRICT — NON-NEGOTIABLE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON in this exact format:

{{
  "query": "SELECT ..."
}}

- Do NOT return raw SQL.
- Do NOT wrap output in ```sql blocks.
- Do NOT include explanations or extra text.

Any response that deviates from this format is INVALID.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCHEMA:
{optimized_schema}

SEMANTIC_CONTEXT:
{semantic_context}

USER QUESTION:
{rewritten_question}
""".strip()
    prompt = (
        f"<|im_start|>system\n{system_instruction}\n\n"
        f"<DB_SCHEMA>\n{optimized_schema}\n</DB_SCHEMA>\n\n"
        f"<SEMANTIC_CONTEXT>\n{semantic_context}\n</SEMANTIC_CONTEXT>\n"
        f"<QUESTION>\n{rewritten_question}\n</QUESTION>\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{rewritten_question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    out = model(prompt, max_tokens=1400)
    raw = out["choices"][0]["text"].strip()

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    print("\n=== RAW MODEL OUTPUT ===\n", raw)

    # ---------------- TRY JSON ----------------
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    parsed = None
    if match:
        json_str = match.group()
        try:

            parsed = json.loads(json_str)
            if "query" in parsed:
                parsed["query"] = fix_table_names(parsed["query"])
            update_chat_history(user_question, json.dumps(parsed))
            return parsed
        except json.JSONDecodeError:
            logger.error(f"[DB Pipeline Error] Invalid JSON:\n{json_str}")

    # ---------------- FALLBACK: EXTRACT SQL ----------------
    if parsed is None:
        sql_match = re.search(r"SELECT.*?;", raw, re.DOTALL | re.IGNORECASE)

        if sql_match:
            query = sql_match.group().strip()

            logger.warning("[DB Pipeline] Recovered SQL without JSON")
            query = fix_table_names(query)

            parsed = {"query": query}

            update_chat_history(user_question, json.dumps(parsed))
            return parsed

    # ---------------- TOTAL FAILURE ----------------
    logger.error(f"[DB Pipeline Error] No JSON or SQL found:\n{raw}")
    update_chat_history(user_question, "ERROR: No valid output")

    raise ValueError("Model did not return JSON or SQL")

# -------------------------------
# SQL RETRY FIX
# -------------------------------
def regenerate_sql_with_feedback(user_question, error_msg,optimized_schema, semantic_context):

    system_instruction = f"""
ROLE:
You are a SQL correction agent.

TASK:
Fix the SQL query based on the error.

ERROR:
{error_msg}

RULES:
- Fix ONLY the SQL
- Use schema strictly
- Keep intent same

OUTPUT:
{{
  "query": "<CORRECTED_SQL>"
}}
""".strip()

    prompt = (
        f"<|im_start|>system\n{system_instruction}\n\n"
        f"<DB_SCHEMA>\n{optimized_schema}\n</DB_SCHEMA>\n\n"
        f"<SEMANTIC_CONTEXT>\n{semantic_context}\n</SEMANTIC_CONTEXT>\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{user_question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    out = model(prompt)
    raw = out["choices"][0]["text"]

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())

    return None

# -------------------------------
# INTELLIGENCE PIPELINE
# -------------------------------
def execute_intelligence_pipeline(user_question):

    chat_context = get_chat_context()

    system_instruction = f"""
ROLE:
You are a Government Assistant.

------------------------------------------------
TASK
------------------------------------------------
Analyze the user message and respond appropriately.

------------------------------------------------
RULE 1 — GREETINGS
------------------------------------------------
Example : user messages like "hello" , "hey", "how are you", etc that are usually used in greetings
Respond politely and ask what can you help them with ( in 1 - 2 lines max ).

------------------------------------------------
RULE 2 — ABUSE
------------------------------------------------
Respond EXACTLY:
Please be respectful.

------------------------------------------------
RULE 3 — OTHER THAN DB QUESTIONS or GREETINGS or ABUSE
------------------------------------------------
Respond EXACTLY:
 I can helps you provide insights, KPI reports, and analytics from the existing DBT database.
------------------------------------------------
CONSTRAINTS
------------------------------------------------
- No DB mention
- No pipeline mention

User:
{user_question}
""".strip()

    prompt = f"<|im_start|>system\n{system_instruction}\n<|im_end|>\n<|im_start|>assistant\n"

    out = model(prompt)
    answer = out["choices"][0]["text"].strip()

    update_chat_history(user_question, answer)
    return answer

# -------------------------------
# TEST
# -------------------------------
if __name__ == "__main__":
    q = "how many schemes are there in the financial year 2024-25?"
    print(execute_db_pipeline(q))
    db_result = execute_db_pipeline(q)

    sql = db_result.get("query", "").replace("`", "")
    logger.info(f"QUERY GENERATED: {sql}")
    

    try:
        rows, error = execute_query(sql)

        print("Query executed successfully")
        print(f"Rows: {rows}")

    except Exception as e:
        print(f"Query failed with error: {str(e)}")
