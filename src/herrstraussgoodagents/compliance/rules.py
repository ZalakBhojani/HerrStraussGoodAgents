from __future__ import annotations

import re
from dataclasses import dataclass

# Forbidden phrases — loaded statically; evaluator also reads these from rubric_v1.yaml
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

# Patterns for illegal threat detection (FDCPA §807)
_ILLEGAL_THREAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(arrest|jail|prison|criminal charge)\b", re.IGNORECASE),
    re.compile(r"\bsue\s+you\b", re.IGNORECASE),
    re.compile(r"\b(garnish|seize|levy)\b.*\b(wage|account|asset)\b", re.IGNORECASE),
]

# Patterns for third-party disclosure violation (FDCPA §805)
_THIRD_PARTY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btell\s+(your\s+)?(employer|boss|family|spouse|neighbor)\b", re.IGNORECASE),
    re.compile(r"\bcontact\s+(your\s+)?(employer|family)\b", re.IGNORECASE),
]


@dataclass
class ComplianceResult:
    passed: bool
    violation: str | None = None


def check_forbidden_phrases(text: str) -> ComplianceResult:
    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lower:
            return ComplianceResult(passed=False, violation=f"Forbidden phrase detected: '{phrase}'")
    return ComplianceResult(passed=True)


def check_no_illegal_threats(text: str) -> ComplianceResult:
    for pattern in _ILLEGAL_THREAT_PATTERNS:
        if pattern.search(text):
            return ComplianceResult(
                passed=False,
                violation=f"Potential illegal threat (FDCPA §807): matched '{pattern.pattern}'",
            )
    return ComplianceResult(passed=True)


def check_no_third_party_disclosure(text: str) -> ComplianceResult:
    for pattern in _THIRD_PARTY_PATTERNS:
        if pattern.search(text):
            return ComplianceResult(
                passed=False,
                violation=f"Third-party disclosure violation (FDCPA §805): matched '{pattern.pattern}'",
            )
    return ComplianceResult(passed=True)


def check_all(text: str) -> ComplianceResult:
    """Run all real-time compliance checks. Returns first violation found."""
    for check in (
        check_forbidden_phrases,
        check_no_illegal_threats,
        check_no_third_party_disclosure,
    ):
        result = check(text)
        if not result.passed:
            return result
    return ComplianceResult(passed=True)
