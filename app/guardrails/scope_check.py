# app/guardrails/scope_check.py
# DEFENDS AGAINST: out-of-scope queries
# METHOD:
#   Layer 1 — keyword blocklist (instant)
#   Layer 2 — vector similarity score check
#             if query is too dissimilar from all docs, refuse it

BLOCKED_KEYWORDS = [
    "write malware", "create virus", "hack into", "how to hack",
    "write a script to", "rm -rf", "drop table", "sql injection",
    "base64 decode", "reverse shell", "exploit", "rootkit",
    "write code to", "generate code for",
]

# Similarity distance threshold for ChromaDB (L2 distance)
# Lower = more similar. If best match > this, query is off-topic.
# Tune this based on your documents — start at 1.5, lower if too permissive
SIMILARITY_THRESHOLD = 1.5

class ScopeResult:
    def __init__(self, in_scope: bool, reason: str = ""):
        self.in_scope = in_scope
        self.reason   = reason

def check_scope(user_query: str, vectorstore) -> ScopeResult:
    """
    Two-layer scope check:
    1. Keyword blocklist — fast hardcoded check
    2. Vector similarity — if no doc chunk is close enough, refuse
    """
    # Layer 1: keyword blocklist
    q_lower = user_query.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in q_lower:
            return ScopeResult(
                in_scope=False,
                reason=f"Blocked keyword: '{kw}'"
            )

    # Layer 2: similarity score check
    # similarity_search_with_score returns (doc, score) pairs
    # For ChromaDB with L2: score=0 means identical, higher=less similar
    try:
        results = vectorstore.similarity_search_with_score(user_query, k=1)
        if results:
            top_doc, top_score = results[0]
            if top_score > SIMILARITY_THRESHOLD:
                return ScopeResult(
                    in_scope=False,
                    reason=f"Query not related to documents (distance={top_score:.2f})"
                )
    except Exception as e:
        # Fail open — if score check errors, don't block legitimate queries
        pass

    return ScopeResult(in_scope=True)