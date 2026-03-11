from typing import Literal, List
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: List[str]


class Verifier:
    def verify(self, question: str, answer: str, context_chunks: list[str]) -> VerificationResult:
        if not context_chunks:
            return VerificationResult(
                verdict="unsupported",
                confidence=0.95,
                reason="No evidence context was retrieved.",
                evidence=[],
            )

        answer_lower = answer.lower().strip()
        matched_evidence = []

        for chunk in context_chunks:
            chunk_lower = chunk.lower()
            if any(word in chunk_lower for word in answer_lower.split() if len(word) > 4):
                matched_evidence.append(chunk)

        if not answer_lower:
            return VerificationResult(
                verdict="unsupported",
                confidence=0.99,
                reason="The answer is empty.",
                evidence=[],
            )

        if answer_lower == "insufficient evidence in provided sources.":
            return VerificationResult(
                verdict="unsupported",
                confidence=0.9,
                reason="The system reported insufficient evidence.",
                evidence=[],
            )

        if matched_evidence:
            verdict = "supported"
            confidence = 0.8
            reason = "The answer is grounded in the retrieved context."
            evidence = matched_evidence[:2]
        else:
            verdict = "unsupported"
            confidence = 0.85
            reason = "The answer introduces claims not clearly found in the retrieved context."
            evidence = []

        return VerificationResult(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            evidence=evidence,
        )