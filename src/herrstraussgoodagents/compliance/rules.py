from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Forbidden phrases — deterministic block list (FDCPA + agency policy)
# ---------------------------------------------------------------------------

FORBIDDEN_PHRASES: list[str] = [
    "i'll sue you",
    "we will arrest",
    "you'll go to jail",
    "destroy your credit",
    "tell your employer",
    "tell your family",
    "you're a deadbeat",
    "piece of trash",
    "you should be ashamed",
    "irresponsible",
]

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Rule 2: No illegal threats (FDCPA §807)
_ILLEGAL_THREAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(arrest|jail|prison|criminal charge)\b", re.IGNORECASE),
    re.compile(r"\bsue\s+you\b", re.IGNORECASE),
    re.compile(r"\b(garnish|seize|levy)\b.*\b(wage|account|asset)\b", re.IGNORECASE),
]

# Rule 3: No calling outside 8am–9pm (FDCPA §805(a)(1))
# Catches agent messages that *promise* an out-of-hours call
_OUT_OF_HOURS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(call|contact|reach)\b.{0,30}\b([01]?\d|2[0-3]):[0-5]\d\s*(am|pm)?\b", re.IGNORECASE),
    re.compile(r"\bmidnight\b|\b2\s*am\b|\b3\s*am\b|\b4\s*am\b|\b5\s*am\b|\b6\s*am\b|\b7\s*am\b", re.IGNORECASE),
]

# Rule 4: No abusive/obscene language (FDCPA §806(2))
_ABUSIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(idiot|moron|stupid|loser|worthless|pathetic|scum)\b", re.IGNORECASE),
    re.compile(r"\b(shut\s+up|get\s+lost)\b", re.IGNORECASE),
]

# Rule 5: No false representations about debt amount (FDCPA §807(2))
# Catches explicit "$0 owed" or "you owe nothing" style false claims
_FALSE_REPRESENTATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou\s+owe\s+\$?0\b", re.IGNORECASE),
    re.compile(r"\bno\s+(debt|balance|amount)\s+is\s+owed\b", re.IGNORECASE),
    re.compile(r"\bdebt\s+(has\s+been\s+)?forgiven\b", re.IGNORECASE),
]

# Rule 8: No third-party disclosure (FDCPA §805(b))
_THIRD_PARTY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btell\s+(your\s+)?(employer|boss|family|spouse|neighbor)\b", re.IGNORECASE),
    re.compile(r"\bcontact\s+(your\s+)?(employer|family)\b", re.IGNORECASE),
    re.compile(r"\bnotify\s+(your\s+)?(employer|spouse|neighbor)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ComplianceResult:
    passed: bool
    violation: str | None = None
    compliance_score_cap: float | None = None  # set to 2.0 on violation


_VIOLATION_SCORE_CAP = 2.0


def _fail(violation: str) -> ComplianceResult:
    return ComplianceResult(passed=False, violation=violation, compliance_score_cap=_VIOLATION_SCORE_CAP)


# ---------------------------------------------------------------------------
# Rule 1: Forbidden phrase block list
# ---------------------------------------------------------------------------

def check_forbidden_phrases(text: str) -> ComplianceResult:
    """Block list of phrases that are always impermissible."""
    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return _fail(f"Forbidden phrase: '{phrase}'")
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 2: No illegal threats (FDCPA §807)
# ---------------------------------------------------------------------------

def check_no_illegal_threats(text: str) -> ComplianceResult:
    """Detect arrest/jail/sue threats that haven't actually been initiated."""
    for pattern in _ILLEGAL_THREAT_PATTERNS:
        if pattern.search(text):
            return _fail(
                f"Illegal threat (FDCPA §807): matched '{pattern.pattern}'"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 3: No out-of-hours contact promises (FDCPA §805(a)(1))
# ---------------------------------------------------------------------------

def check_no_out_of_hours_promise(text: str) -> ComplianceResult:
    """Detect agent promising to call at legally restricted hours."""
    for pattern in _OUT_OF_HOURS_PATTERNS:
        if pattern.search(text):
            return _fail(
                f"Out-of-hours contact promise (FDCPA §805): matched '{pattern.pattern}'"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 4: No abusive / obscene language (FDCPA §806(2))
# ---------------------------------------------------------------------------

def check_no_abusive_language(text: str) -> ComplianceResult:
    """Detect abusive or demeaning language."""
    for pattern in _ABUSIVE_PATTERNS:
        if pattern.search(text):
            return _fail(
                f"Abusive language (FDCPA §806): matched '{pattern.pattern}'"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 5: No false representations about debt (FDCPA §807(2))
# ---------------------------------------------------------------------------

def check_no_false_debt_representation(text: str) -> ComplianceResult:
    """Detect explicit false statements about the debt amount or status."""
    for pattern in _FALSE_REPRESENTATION_PATTERNS:
        if pattern.search(text):
            return _fail(
                f"False debt representation (FDCPA §807(2)): matched '{pattern.pattern}'"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 6: Cease-communication acknowledgment
# ---------------------------------------------------------------------------

_CEASE_REQUEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bstop\s+(calling|contacting|all\s+contact)\b", re.IGNORECASE),
    re.compile(r"\bcease\s+(and\s+desist|communication|contact)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(call|contact)\b", re.IGNORECASE),
    re.compile(r"\bremove\s+(me|my\s+number)\b", re.IGNORECASE),
]

_CEASE_ACKNOWLEDGMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bhonor\b|\brespect\b|\bnote\b|\bdocument\b", re.IGNORECASE),
    re.compile(r"\bwill\s+not\s+(call|contact)\b", re.IGNORECASE),
    re.compile(r"\byour\s+request\b", re.IGNORECASE),
]


def check_cease_acknowledgment(text: str, borrower_text: str = "") -> ComplianceResult:
    """If borrower requested cease-contact, agent response must acknowledge it.

    This check is only relevant when called with the borrower's prior message.
    When borrower_text is empty, it passes automatically.
    """
    if not borrower_text:
        return ComplianceResult(passed=True)
    borrower_lower = borrower_text.lower()
    agent_lower = text.lower()
    if any(p.search(borrower_lower) for p in _CEASE_REQUEST_PATTERNS):
        if not any(p.search(agent_lower) for p in _CEASE_ACKNOWLEDGMENT_PATTERNS):
            return _fail(
                "Agent failed to acknowledge cease-communication request (FDCPA §805(c))"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 7: Debt validation notice availability (FDCPA §809)
# ---------------------------------------------------------------------------

_VALIDATION_REQUEST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(verify|validate|proof|prove)\s+(the\s+)?(debt|amount|balance)\b", re.IGNORECASE),
    re.compile(r"\bdebt\s+validation\b", re.IGNORECASE),
    re.compile(r"\bwho\s+do\s+you\s+represent\b", re.IGNORECASE),
]

_VALIDATION_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bvalidation\s+notice\b", re.IGNORECASE),
    re.compile(r"\bwritten\s+(notice|confirmation|verification)\b", re.IGNORECASE),
    re.compile(r"\bsend\s+you\b|\bprovide\s+you\b|\bforward\b", re.IGNORECASE),
    re.compile(r"\b30\s+day\b|\bthirty\s+day\b", re.IGNORECASE),
]


def check_debt_validation_response(text: str, borrower_text: str = "") -> ComplianceResult:
    """If borrower requests debt validation, agent must offer to provide it."""
    if not borrower_text:
        return ComplianceResult(passed=True)
    borrower_lower = borrower_text.lower()
    agent_lower = text.lower()
    if any(p.search(borrower_lower) for p in _VALIDATION_REQUEST_PATTERNS):
        if not any(p.search(agent_lower) for p in _VALIDATION_RESPONSE_PATTERNS):
            return _fail(
                "Agent failed to offer debt validation notice when requested (FDCPA §809)"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Rule 8: No third-party disclosure (FDCPA §805(b))
# ---------------------------------------------------------------------------

def check_no_third_party_disclosure(text: str) -> ComplianceResult:
    """Detect attempts to discuss debt with third parties."""
    for pattern in _THIRD_PARTY_PATTERNS:
        if pattern.search(text):
            return _fail(
                f"Third-party disclosure (FDCPA §805(b)): matched '{pattern.pattern}'"
            )
    return ComplianceResult(passed=True)


# ---------------------------------------------------------------------------
# Aggregate — runs all context-free checks (no borrower_text needed)
# ---------------------------------------------------------------------------

def check_all(text: str) -> ComplianceResult:
    """Run all real-time, context-free compliance checks.

    Returns the first violation found; passes if all clear.
    Rules 6 and 7 require borrower context and are called separately
    from BaseAgent.generate() where the prior borrower message is available.
    """
    for check in (
        check_forbidden_phrases,
        check_no_illegal_threats,
        check_no_out_of_hours_promise,
        check_no_abusive_language,
        check_no_false_debt_representation,
        check_no_third_party_disclosure,
    ):
        result = check(text)
        if not result.passed:
            return result
    return ComplianceResult(passed=True)


def check_all_with_context(text: str, borrower_text: str) -> ComplianceResult:
    """Run all checks including context-sensitive rules (6 + 7).

    Call this from agents that have access to the prior borrower message.
    """
    result = check_all(text)
    if not result.passed:
        return result
    result = check_cease_acknowledgment(text, borrower_text)
    if not result.passed:
        return result
    return check_debt_validation_response(text, borrower_text)
