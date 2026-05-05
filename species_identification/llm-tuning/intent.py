"""
Intent classification for RoboRanger queries.

Classifies a user query into one of a fixed set of intents using cosine
similarity against precomputed prototype embeddings. No LLM call required.

Intents drive:
  - which prompt template the generator uses
  - which blurb fields are emphasized (e.g. DANGER -> venom/hazards)
  - optional retrieval boosting (filter/rerank by intent metadata if present)

Confidence scoring:
  - margin = top_score - second_score
  - high confidence  : top >= 0.55 AND margin >= 0.08
  - medium confidence: top >= 0.45
  - low / OTHER      : everything else -> caller should treat as OTHER

Tune the thresholds against a labeled query set; the defaults are starting
points, not gospel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class Intent(str, Enum):
    DESCRIPTION = "description"      # "what does it look like", "tell me about it"
    DANGER = "danger"                # "is it venomous", "will it hurt me", "is it safe"
    DIET = "diet"                    # "what does it eat", "is it a predator"
    HABITAT = "habitat"              # "where does it live", "what's its range"
    BEHAVIOR = "behavior"            # "is it aggressive", "is it nocturnal", "how does it hunt"
    IDENTIFICATION = "identification"  # "how do I tell it apart from", "what are key features"
    CONSERVATION = "conservation"    # "is it endangered", "is it protected"
    OTHER = "other"                  # fallback


# Prototype phrases per intent. Keep these short, varied, and written the way
# a real user would phrase the question. 4-8 prototypes per intent is the
# sweet spot — more doesn't help much and dilutes the centroid.
PROTOTYPES: dict[Intent, list[str]] = {
    Intent.DESCRIPTION: [
        "what does this animal look like",
        "describe this species",
        "tell me about it",
        "what is this",
        "give me an overview",
        "what kind of animal is this",
    ],
    Intent.DANGER: [
        "is it venomous",
        "is it poisonous",
        "is it dangerous to humans",
        "will it hurt me",
        "is it safe to approach",
        "can it bite",
        "should I be worried about this animal",
    ],
    Intent.DIET: [
        "what does it eat",
        "what is its diet",
        "is it a predator",
        "is it a herbivore",
        "what does it hunt",
        "what does it feed on",
    ],
    Intent.HABITAT: [
        "where does it live",
        "what is its habitat",
        "what is its range",
        "where can I find it",
        "what environment does it prefer",
        "is it native to this area",
    ],
    Intent.BEHAVIOR: [
        "is it aggressive",
        "is it nocturnal",
        "how does it hunt",
        "how does it behave",
        "is it active during the day",
        "does it live in groups",
        "how does it defend itself",
    ],
    Intent.IDENTIFICATION: [
        "how do I tell it apart from similar species",
        "what are its key identifying features",
        "how can I identify this",
        "what should I look for to identify it",
        "what makes this species distinctive",
    ],
    Intent.CONSERVATION: [
        "is it endangered",
        "is it protected",
        "what is its conservation status",
        "is it threatened",
        "is it rare",
    ],
}


@dataclass
class IntentResult:
    intent: Intent
    confidence: str              # "high", "medium", "low"
    score: float                 # cosine similarity to top intent
    margin: float                # gap between top and second
    all_scores: dict[Intent, float]

    @property
    def is_confident(self) -> bool:
        return self.confidence in ("high", "medium")


class IntentClassifier:
    """
    Embedding-based intent classifier.

    Pass in your existing bge-small-en-v1.5 embedder so we don't load a second
    model. The embedder must expose `.encode(list[str]) -> np.ndarray` with
    L2-normalized rows (bge does this when normalize_embeddings=True).
    """

    # Tuned from real bge-small-en-v1.5 eval data:
    #   - Correct in-domain queries score 0.68-0.95 against their centroid.
    #   - Margins on real queries cluster around 0.04-0.06, not 0.08+.
    # Note: this classifier no longer needs to reject off-topic queries —
    # WildlifeGate handles that upstream. So thresholds here are about
    # confidence labeling, not gating, and "low" confidence cases should
    # still proceed to retrieval (they just signal uncertainty downstream).
    HIGH_SCORE_THRESHOLD = 0.70
    HIGH_MARGIN_THRESHOLD = 0.04
    MEDIUM_SCORE_THRESHOLD = 0.60

    def __init__(self, embedder, *, normalize: bool = True) -> None:
        self.embedder = embedder
        self.normalize = normalize
        self._intents: list[Intent] = []
        self._centroids: np.ndarray | None = None
        self._build_centroids()

    def _build_centroids(self) -> None:
        intents: list[Intent] = []
        centroids: list[np.ndarray] = []
        for intent, prototypes in PROTOTYPES.items():
            embs = self.embedder.encode(prototypes)
            embs = np.asarray(embs, dtype=np.float32)
            if self.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embs = embs / norms
            centroid = embs.mean(axis=0)
            # Re-normalize centroid so dot product == cosine similarity.
            n = np.linalg.norm(centroid)
            if n > 0:
                centroid = centroid / n
            intents.append(intent)
            centroids.append(centroid)
        self._intents = intents
        self._centroids = np.stack(centroids, axis=0)  # (num_intents, dim)

    def classify(self, query: str) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(
                intent=Intent.OTHER,
                confidence="low",
                score=0.0,
                margin=0.0,
                all_scores={i: 0.0 for i in self._intents},
            )

        q_emb = np.asarray(self.embedder.encode([query]), dtype=np.float32)[0]
        if self.normalize:
            n = np.linalg.norm(q_emb)
            if n > 0:
                q_emb = q_emb / n

        assert self._centroids is not None
        scores = self._centroids @ q_emb  # (num_intents,)
        order = np.argsort(-scores)
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx

        top_score = float(scores[top_idx])
        second_score = float(scores[second_idx])
        margin = top_score - second_score

        if (
            top_score >= self.HIGH_SCORE_THRESHOLD
            and margin >= self.HIGH_MARGIN_THRESHOLD
        ):
            confidence = "high"
            intent = self._intents[top_idx]
        elif top_score >= self.MEDIUM_SCORE_THRESHOLD:
            confidence = "medium"
            intent = self._intents[top_idx]
        else:
            confidence = "low"
            intent = Intent.OTHER

        all_scores = {
            self._intents[i]: float(scores[i]) for i in range(len(self._intents))
        }
        return IntentResult(
            intent=intent,
            confidence=confidence,
            score=top_score,
            margin=margin,
            all_scores=all_scores,
        )


# ---- quick smoke test (runs only when invoked directly) ----
if __name__ == "__main__":
    # Dummy embedder for offline testing — replace with bge-small at runtime.
    class _HashEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            rng = np.random.default_rng(0)
            out = []
            for t in texts:
                seed = abs(hash(t)) % (2**32)
                r = np.random.default_rng(seed)
                out.append(r.standard_normal(384).astype(np.float32))
            return np.stack(out)

    clf = IntentClassifier(_HashEmbedder())
    for q in [
        "is this rattlesnake venomous?",
        "what does a coyote eat?",
        "where do mule deer live?",
        "tell me about this bird",
        "asdf qwerty",
    ]:
        r = clf.classify(q)
        print(f"{q!r:50s} -> {r.intent.value:14s} {r.confidence:6s} "
              f"score={r.score:+.3f} margin={r.margin:+.3f}")