"""
Wildlife topic gate for RoboRanger.

Runs BEFORE intent classification. Decides whether a query is about wildlife
at all. If not, the pipeline short-circuits to a fixed "I can only help with
wildlife questions" response — no retrieval, no LLM call.

Why this is a separate component:
  Intent classification answers "what kind of wildlife question is this?"
  This gate answers "is this about wildlife at all?" Different questions,
  different prototypes, different threshold philosophy.

Mechanism:
  Two centroids — one for in-domain wildlife queries, one for off-topic
  park queries. Classify by which is closer (relative comparison, not
  absolute threshold). The relative approach matters because off-topic
  queries like "where is the bathroom" score high against wildlife intents
  in absolute terms — they only look off-topic when compared against an
  off-topic centroid directly.

Tuning philosophy:
  Bias toward accept on borderline cases (cost of refusing a valid query
  is annoying; cost of confidently hallucinating an answer to an off-topic
  query is worse, but rare). On ambiguous cases (similar scores on both
  sides), default to reject — the LLM has nothing useful to say about
  bathroom locations, so passing through is pure downside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# In-domain prototypes: things visitors actually ask about wildlife. Mix
# question forms (what/where/how/is it) and topics (appearance, danger,
# behavior, etc.) so the centroid covers the full intent space.
WILDLIFE_PROTOTYPES = [
    "what is this animal",
    "is it venomous",
    "what does it eat",
    "where does it live",
    "is it nocturnal",
    "how can I identify it",
    "is it endangered",
    "is this dangerous",
    "tell me about this species",
    "is it native to california",
    "does it bite",
    "what does it look like",
    "is this a predator",
    "how does it hunt",
    "is it aggressive",
    # Added after first eval — query forms that got false-rejected:
    "give me an overview of this animal",
    "is it safe to approach",
    "is it from around here",
    "where would I find this species",
    "what does it do during the day",
    "what features should I look for",
    "tell me more",
    "what should I know about it",
]

# Off-topic prototypes: park-visitor queries that AREN'T about wildlife.
# Important: anchor these on park-logistics nouns (bathroom, parking,
# trail, map, hours) rather than generic question structures. The previous
# version absorbed too much "where is X" / "is X" signal and started
# matching wildlife queries that happened to share those structures.
OFF_TOPIC_PROTOTYPES = [
    "where is the nearest bathroom",
    "where is the parking lot",
    "where is the visitor center",
    "what time does the park close",
    "what time does the visitor center open",
    "is the trail closed today",
    "how far is it to the trailhead",
    "where can I buy a map of the park",
    "is there a gift shop nearby",
    "where can I get water or food",
    "when does the next ranger tour start",
    "is there cell service here",
    "how much is the entrance fee",
    "where can I park my car",
    "is camping allowed in the park",
    # Chitchat / non-questions:
    "the weather is nice today",
    "thanks for the help",
    "hello there",
    "good morning",
    # Junk:
    "asdfasdf",
    "test test test",
]


@dataclass
class GateResult:
    in_domain: bool
    confidence: str          # "high", "medium", "low"
    wildlife_score: float
    off_topic_score: float
    margin: float            # wildlife_score - off_topic_score (signed)

    @property
    def reject(self) -> bool:
        return not self.in_domain


class WildlifeGate:
    """
    Binary classifier: is this query about wildlife, or about park logistics
    / chitchat / nonsense?

    Threshold philosophy (revised after first eval):
      - Any positive margin -> accept. If the wildlife centroid wins at all,
        let the query through. False rejects are user-visible and annoying;
        false accepts get caught downstream by the LLM's "I don't know"
        instruction in the system prompt.
      - margin >= ACCEPT_MARGIN  -> accept, high confidence
      - 0 <= margin < ACCEPT     -> accept, medium confidence
      - -REJECT < margin < 0     -> reject, low confidence (ambiguous)
      - margin <= -REJECT_MARGIN -> reject, high confidence

    The earlier symmetric design was too aggressive: queries near the
    boundary got rejected even when the wildlife side was winning. With
    real bge-small embeddings, even a +0.01 margin is a meaningful signal.
    """

    ACCEPT_MARGIN = 0.05      # confident accept above this
    REJECT_MARGIN = 0.03      # confident reject below the negation of this

    def __init__(self, embedder, *, normalize: bool = True) -> None:
        self.embedder = embedder
        self.normalize = normalize
        self._wildlife_centroid: np.ndarray | None = None
        self._off_topic_centroid: np.ndarray | None = None
        self._build_centroids()

    def _build_centroids(self) -> None:
        self._wildlife_centroid = self._centroid_of(WILDLIFE_PROTOTYPES)
        self._off_topic_centroid = self._centroid_of(OFF_TOPIC_PROTOTYPES)

    def _centroid_of(self, texts: Sequence[str]) -> np.ndarray:
        embs = np.asarray(self.embedder.encode(list(texts)), dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embs = embs / norms
        c = embs.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        return c

    def check(self, query: str) -> GateResult:
        if not query or not query.strip():
            return GateResult(False, "high", 0.0, 0.0, 0.0)

        q_emb = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        if self.normalize:
            n = np.linalg.norm(q_emb)
            if n > 0:
                q_emb = q_emb / n

        assert self._wildlife_centroid is not None
        assert self._off_topic_centroid is not None
        wildlife_score = float(self._wildlife_centroid @ q_emb)
        off_topic_score = float(self._off_topic_centroid @ q_emb)
        margin = wildlife_score - off_topic_score

        # Degenerate case: query embedding has no meaningful similarity to
        # either centroid (zero vector, completely orthogonal text). Reject.
        if wildlife_score < 0.05 and off_topic_score < 0.05:
            return GateResult(False, "low", wildlife_score, off_topic_score, margin)

        if margin >= self.ACCEPT_MARGIN:
            return GateResult(True, "high", wildlife_score, off_topic_score, margin)
        if margin >= 0:
            # Wildlife side wins, but not strongly. Let it through.
            return GateResult(True, "medium", wildlife_score, off_topic_score, margin)
        if margin > -self.REJECT_MARGIN:
            # Off-topic side wins narrowly. Reject, but flag low confidence.
            return GateResult(False, "low", wildlife_score, off_topic_score, margin)
        # Off-topic side wins clearly.
        return GateResult(False, "high", wildlife_score, off_topic_score, margin)


# ---- smoke test (with hash embedder, just plumbing) ----
if __name__ == "__main__":
    class _HashEmbedder:
        def encode(self, texts):
            out = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                r = np.random.default_rng(seed)
                out.append(r.standard_normal(384).astype(np.float32))
            return np.stack(out)

    gate = WildlifeGate(_HashEmbedder())
    for q in [
        "is this snake venomous?",
        "where is the bathroom",
        "what time does the park close",
        "what does a coyote eat",
    ]:
        r = gate.check(q)
        verdict = "ACCEPT" if r.in_domain else "REJECT"
        print(f"{q!r:50s} -> {verdict:6s} ({r.confidence:6s}) "
              f"wild={r.wildlife_score:+.3f} off={r.off_topic_score:+.3f} "
              f"margin={r.margin:+.3f}")