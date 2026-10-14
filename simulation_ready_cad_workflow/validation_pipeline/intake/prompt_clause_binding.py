from __future__ import annotations

"""Conservative syntactic binding for acceptance clauses.

The immutable prompt remains the authority.  A clause may quote one literal
span, join explicit source spans with an explicit connector, omit a short
coordinated sibling property, or resolve one tightly bounded role pronoun.
No fuzzy similarity, unordered token matching, or semantic model is used.
"""

import re


_TOKEN_RE = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SENSITIVE_ELISION_TOKENS = frozenset(
    {
        "above",
        "back",
        "backward",
        "below",
        "either",
        "equal",
        "equals",
        "forward",
        "least",
        "may",
        "most",
        "must",
        "negative",
        "never",
        "no",
        "not",
        "only",
        "or",
        "positive",
        "reverse",
        "shall",
        "should",
        "under",
        "without",
    }
)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(item.casefold() for item in _TOKEN_RE.findall(value))


def _find_spans(
    haystack: tuple[str, ...],
    needle: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    if not needle or len(needle) > len(haystack):
        return ()
    width = len(needle)
    return tuple(
        (index, index + width)
        for index in range(len(haystack) - width + 1)
        if haystack[index : index + width] == needle
    )


def _ordered_explicit_spans(prompt: str, clause: str) -> bool:
    connector: str | None = None
    fragments: list[str]
    if ";" in clause:
        fragments = clause.split(";")
        connector = ";"
    elif re.search(r"\bso\s+that\b", clause, flags=re.IGNORECASE):
        fragments = re.split(
            r"\bso\s+that\b",
            clause,
            flags=re.IGNORECASE,
        )
        connector = "so that"
    else:
        return False
    if len(fragments) != 2:
        return False
    fragment_tokens = tuple(_tokens(item) for item in fragments)
    if any(len(item) < 2 for item in fragment_tokens):
        return False
    prompt_tokens = _tokens(prompt)
    left_spans = _find_spans(prompt_tokens, fragment_tokens[0])
    right_spans = _find_spans(prompt_tokens, fragment_tokens[1])
    for left in left_spans:
        for right in right_spans:
            if left[1] > right[0]:
                continue
            if connector == "so that" and (
                "so",
                "that",
            ) not in tuple(
                prompt_tokens[index : index + 2]
                for index in range(left[1], max(left[1], right[0] - 1))
            ):
                continue
            return True
    return False


def _coordinated_property_elision(prompt: str, clause: str) -> bool:
    clause_tokens = _tokens(clause)
    if len(clause_tokens) < 4:
        return False
    for sentence in _SENTENCE_RE.split(prompt):
        source = _tokens(sentence)
        for split in range(1, len(clause_tokens)):
            prefix = clause_tokens[:split]
            suffix = clause_tokens[split:]
            for prefix_start, prefix_end in _find_spans(source, prefix):
                for gap_width in range(1, 5):
                    suffix_start = prefix_end + gap_width
                    suffix_end = suffix_start + len(suffix)
                    if source[suffix_start:suffix_end] != suffix:
                        continue
                    insertion = source[prefix_end:suffix_start]
                    if "and" not in insertion:
                        continue
                    if any(
                        token in _SENSITIVE_ELISION_TOKENS
                        or any(character.isdigit() for character in token)
                        for token in insertion
                    ):
                        continue
                    # The matched source is one contiguous sentence span;
                    # text before/after it is deliberately irrelevant.
                    if prefix_start >= 0:
                        return True
    return False


def _regular_ing_pair(left: str, right: str) -> bool:
    """Accept only a single regular base/``-ing`` spelling pair."""

    def bases(value: str) -> frozenset[str]:
        if not value.endswith("ing") or len(value) <= 4:
            return frozenset()
        stem = value[:-3]
        candidates = {stem, f"{stem}e"}
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            candidates.add(stem[:-1])
        return frozenset(candidates)

    return right in bases(left) or left in bases(right)


def _context_tokens_match(
    clause_tokens: tuple[str, ...],
    source_tokens: tuple[str, ...],
) -> bool:
    if len(clause_tokens) != len(source_tokens):
        return False
    differences = [
        (left, right)
        for left, right in zip(clause_tokens, source_tokens)
        if left != right
    ]
    return len(differences) <= 1 and all(
        _regular_ing_pair(left, right) for left, right in differences
    )


def _role_coreference_while(
    prompt: str,
    clause: str,
    *,
    target_role: str | None,
    input_role: str | None,
) -> bool:
    match = re.fullmatch(
        r"\s*(?P<claim>.+?)\s+while\s+(?P<context>.+?)\s*",
        clause,
        flags=re.IGNORECASE,
    )
    if match is None or not target_role:
        return False
    claim_tokens = _tokens(match.group("claim"))
    context_tokens = _tokens(match.group("context"))
    role_tokens = _tokens(target_role)
    if len(claim_tokens) < len(role_tokens) + 2 or len(context_tokens) < 2:
        return False
    role_spans = _find_spans(claim_tokens, role_tokens)
    if len(role_spans) != 1:
        return False
    role_start, role_end = role_spans[0]
    article_start = role_start - 1
    if article_start < 0 or claim_tokens[article_start] != "the":
        return False
    prompt_role = ("it",) if len(role_tokens) == 1 else ("it",)
    claim_with_pronoun = (
        claim_tokens[:article_start] + prompt_role + claim_tokens[role_end:]
    )
    input_tokens = set(_tokens(input_role or ""))
    forbidden_gap = {
        "no",
        "not",
        "never",
        "without",
        *input_tokens,
    }
    for sentence in _SENTENCE_RE.split(prompt):
        if ":" not in sentence:
            continue
        scope, detail = sentence.split(":", 1)
        scope_tokens = _tokens(scope)
        detail_tokens = _tokens(detail)
        context_matches = any(
            _context_tokens_match(
                context_tokens,
                scope_tokens[start : start + len(context_tokens)],
            )
            for start in range(
                max(0, len(scope_tokens) - len(context_tokens) + 1)
            )
        )
        if not context_matches:
            continue
        claim_spans = _find_spans(detail_tokens, claim_with_pronoun)
        for claim_start, _claim_end in claim_spans:
            antecedents = [
                span
                for span in _find_spans(detail_tokens, role_tokens)
                if span[1] <= claim_start and claim_start - span[1] <= 12
            ]
            if len(antecedents) != 1:
                continue
            gap = detail_tokens[antecedents[0][1] : claim_start]
            if not any(token in forbidden_gap for token in gap):
                return True
    return False


def prompt_clause_is_bound(
    prompt: str,
    clause: str,
    *,
    target_role: str | None = None,
    input_role: str | None = None,
) -> bool:
    """Return whether ``clause`` has bounded syntactic evidence in ``prompt``."""

    normalized_prompt = " ".join(prompt.casefold().split())
    normalized_clause = " ".join(clause.casefold().split())
    if normalized_clause in normalized_prompt:
        return True
    return (
        _ordered_explicit_spans(prompt, clause)
        or _coordinated_property_elision(prompt, clause)
        or _role_coreference_while(
            prompt,
            clause,
            target_role=target_role,
            input_role=input_role,
        )
    )
