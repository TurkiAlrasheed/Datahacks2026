"""
Unit tests for intent.py and prompt_builder.py.

Uses pytest. Run with:
    pytest test_units.py -v

These tests don't need the real embedder — they verify plumbing, edge cases,
and prompt-assembly correctness. The intent eval (eval_intent.py) is the
quality test; this file is the regression test.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys

sys.path.insert(1, "../species_identification/llm-tuning")
from intent import Intent, IntentClassifier, IntentResult, PROTOTYPES
from prompt_builder import (
    Blurb,
    Chunk,
    MAX_BLURB_CHARS,
    MAX_CHARS_PER_CHUNK,
    MAX_CHUNKS,
    SYSTEM_PROMPT,
    _danger_directive,
    _format_danger_verdict,
    _question_polarity,
    build_messages,
    build_prompt,
)
from wildlife_gate import (
    OFF_TOPIC_PROTOTYPES,
    WILDLIFE_PROTOTYPES,
    WildlifeGate,
)


# ---------------------------------------------------------------------------
# Test embedder: returns a fixed, controllable vector per text. Lets us craft
# queries that land arbitrarily close to a chosen intent's centroid.
# ---------------------------------------------------------------------------
class StubEmbedder:
    """Returns a unit vector pointing at a chosen 'intent direction'."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        # One basis direction per intent, deterministic order.
        self.intent_dirs: dict[Intent, np.ndarray] = {}
        for i, intent in enumerate(PROTOTYPES.keys()):
            v = np.zeros(dim, dtype=np.float32)
            v[i] = 1.0
            self.intent_dirs[intent] = v
        # Two extra directions for the wildlife gate.
        n_intents = len(self.intent_dirs)
        self.wildlife_dir = np.zeros(dim, dtype=np.float32)
        self.wildlife_dir[n_intents] = 1.0
        self.off_topic_dir = np.zeros(dim, dtype=np.float32)
        self.off_topic_dir[n_intents + 1] = 1.0

        # Mapping from prototype text -> direction vector.
        self.proto_to_intent: dict[str, Intent] = {}
        for intent, protos in PROTOTYPES.items():
            for p in protos:
                self.proto_to_intent[p] = intent

    def encode(self, texts):
        out = []
        for t in texts:
            if t in self.proto_to_intent:
                out.append(self.intent_dirs[self.proto_to_intent[t]])
            elif t in WILDLIFE_PROTOTYPES:
                out.append(self.wildlife_dir)
            elif t in OFF_TOPIC_PROTOTYPES:
                out.append(self.off_topic_dir)
            elif t.startswith("__intent__:"):
                # Special test syntax: "__intent__:DANGER" -> pure DANGER vec
                name = t.split(":", 1)[1]
                out.append(self.intent_dirs[Intent[name]])
            elif t == "__wildlife__":
                out.append(self.wildlife_dir)
            elif t == "__off_topic__":
                out.append(self.off_topic_dir)
            elif t.startswith("__mix__:"):
                # "__mix__:DANGER:0.8:DIET:0.2" -> mixture
                parts = t.split(":")[1:]
                v = np.zeros(self.dim, dtype=np.float32)
                for i in range(0, len(parts), 2):
                    v += float(parts[i + 1]) * self.intent_dirs[Intent[parts[i]]]
                out.append(v)
            else:
                # Unknown text -> zero vector (will produce score 0 on all).
                out.append(np.zeros(self.dim, dtype=np.float32))
        return np.stack(out).astype(np.float32)


@pytest.fixture
def stub_clf():
    return IntentClassifier(StubEmbedder())


# ===========================================================================
# Intent classifier tests
# ===========================================================================

class TestIntentClassifier:

    def test_pure_intent_query_classifies_correctly(self, stub_clf):
        for intent in [Intent.DANGER, Intent.DIET, Intent.HABITAT,
                       Intent.BEHAVIOR, Intent.IDENTIFICATION,
                       Intent.CONSERVATION, Intent.DESCRIPTION]:
            r = stub_clf.classify(f"__intent__:{intent.name}")
            assert r.intent == intent, f"failed on {intent}"
            assert r.confidence == "high"
            assert r.score > 0.9

    def test_empty_query_returns_other_low(self, stub_clf):
        r = stub_clf.classify("")
        assert r.intent == Intent.OTHER
        assert r.confidence == "low"
        assert r.score == 0.0

    def test_whitespace_query_returns_other_low(self, stub_clf):
        r = stub_clf.classify("   \n\t  ")
        assert r.intent == Intent.OTHER
        assert r.confidence == "low"

    def test_zero_vector_query_falls_to_other(self, stub_clf):
        # Unknown text -> StubEmbedder returns zero vector -> all scores zero.
        r = stub_clf.classify("totally unrecognized gibberish")
        assert r.intent == Intent.OTHER
        assert r.confidence == "low"

    def test_ambiguous_query_low_margin_drops_confidence(self, stub_clf):
        # 50/50 mix between DANGER and DIET. Top score is high but margin
        # is zero, so it should NOT be high-confidence.
        r = stub_clf.classify("__mix__:DANGER:0.5:DIET:0.5")
        assert r.confidence != "high"

    def test_clear_winner_with_margin_is_high(self, stub_clf):
        r = stub_clf.classify("__mix__:DANGER:0.9:DIET:0.1")
        assert r.intent == Intent.DANGER
        assert r.confidence == "high"

    def test_all_scores_includes_every_intent(self, stub_clf):
        r = stub_clf.classify("__intent__:DANGER")
        # Every classifier-known intent should appear in all_scores.
        # (OTHER is not a prototype-bearing intent, so it won't be there.)
        for intent in PROTOTYPES.keys():
            assert intent in r.all_scores

    def test_centroids_are_unit_length(self, stub_clf):
        norms = np.linalg.norm(stub_clf._centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_score_is_bounded(self, stub_clf):
        r = stub_clf.classify("__intent__:DANGER")
        # Cosine sim between unit vectors is in [-1, 1].
        for s in r.all_scores.values():
            assert -1.001 <= s <= 1.001


# ===========================================================================
# Wildlife gate tests
# ===========================================================================

@pytest.fixture
def stub_gate():
    return WildlifeGate(StubEmbedder())


class TestWildlifeGate:

    def test_pure_wildlife_query_accepts(self, stub_gate):
        r = stub_gate.check("__wildlife__")
        assert r.in_domain is True
        assert r.confidence == "high"
        assert r.margin > stub_gate.ACCEPT_MARGIN

    def test_pure_off_topic_query_rejects(self, stub_gate):
        r = stub_gate.check("__off_topic__")
        assert r.in_domain is False
        assert r.confidence == "high"
        assert r.margin < -stub_gate.REJECT_MARGIN

    def test_empty_query_rejects(self, stub_gate):
        r = stub_gate.check("")
        assert r.in_domain is False

    def test_whitespace_query_rejects(self, stub_gate):
        r = stub_gate.check("   ")
        assert r.in_domain is False

    def test_unknown_query_rejects_as_ambiguous(self, stub_gate):
        # Zero vector against both centroids -> margin = 0 -> ambiguous reject.
        r = stub_gate.check("totally unrecognized")
        assert r.in_domain is False
        assert r.confidence == "low"

    def test_reject_property(self, stub_gate):
        r_accept = stub_gate.check("__wildlife__")
        r_reject = stub_gate.check("__off_topic__")
        assert r_accept.reject is False
        assert r_reject.reject is True

    def test_centroids_are_unit_length(self, stub_gate):
        for c in (stub_gate._wildlife_centroid, stub_gate._off_topic_centroid):
            assert c is not None
            np.testing.assert_allclose(np.linalg.norm(c), 1.0, atol=1e-5)

    def test_margin_is_signed_difference(self, stub_gate):
        r = stub_gate.check("__wildlife__")
        np.testing.assert_allclose(
            r.margin, r.wildlife_score - r.off_topic_score, atol=1e-5
        )




@pytest.fixture
def sample_blurb():
    return Blurb(
        common_name="Western Fence Lizard",
        scientific_name="Sceloporus occidentalis",
        appearance="Small spiny lizard with blue belly patches in males. Spiny scales, gray-brown body.",
        size="6-9 cm body length",
        habitat="Rocks, fences, woodpiles across the western US.",
        diet="Insects and small arthropods.",
        behavior="Active in daytime, basks on sunny surfaces.",
        dangerous_to_humans="no",
        dangerous_to_pets="no",
        notable="Their blood kills Lyme disease bacteria in tick guts.",
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(text="Western fence lizards reduce Lyme disease prevalence "
                   "because their blood kills Borrelia in tick guts.",
              score=0.72),
        Chunk(text="They are sometimes called 'blue-bellies' for the blue "
                   "patches on the underside of males.", score=0.65),
    ]


def _ir(intent: Intent, confidence: str = "high") -> IntentResult:
    return IntentResult(
        intent=intent,
        confidence=confidence,
        score=0.7,
        margin=0.15,
        all_scores={},
    )


class TestPromptBuilder:

    def test_messages_have_system_and_user(self, sample_blurb, sample_chunks):
        msgs = build_messages(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=sample_chunks,
            intent_result=_ir(Intent.DIET),
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_relevance_instruction_is_in_system_prompt(self):
        # The relevance gate is the load-bearing instruction. If this string
        # changes accidentally, that's a bug.
        assert "snippets do not address the question" in SYSTEM_PROMPT.lower()
        assert "species facts" in SYSTEM_PROMPT.lower()

    def test_user_content_includes_query(self, sample_blurb, sample_chunks):
        msgs = build_messages(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=sample_chunks,
            intent_result=_ir(Intent.DIET),
        )
        assert "what does it eat?" in msgs[1]["content"]

    def test_intent_specific_field_leads_blurb(self, sample_blurb,
                                                sample_chunks):
        # DANGER intent puts the VERDICT block BEFORE the appearance line.
        msgs = build_messages(
            query="is it dangerous?",
            blurb=sample_blurb,
            chunks=sample_chunks,
            intent_result=_ir(Intent.DANGER),
        )
        content = msgs[1]["content"]
        verdict_pos = content.find("VERDICT:")
        appearance_pos = content.find("- appearance:")
        assert verdict_pos > 0
        assert appearance_pos > 0
        assert verdict_pos < appearance_pos

    def test_diet_intent_leads_with_diet(self, sample_blurb, sample_chunks):
        msgs = build_messages(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=sample_chunks,
            intent_result=_ir(Intent.DIET),
        )
        content = msgs[1]["content"]
        diet_pos = content.find("- diet:")
        appearance_pos = content.find("- appearance:")
        assert diet_pos < appearance_pos

    def test_empty_chunks_produces_explicit_marker(self, sample_blurb):
        msgs = build_messages(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=[],
            intent_result=_ir(Intent.DIET),
        )
        assert "no snippets retrieved" in msgs[1]["content"]

    def test_chunks_are_capped(self, sample_blurb):
        many = [Chunk(text=f"chunk number {i}", score=0.7)
                for i in range(MAX_CHUNKS + 5)]
        msgs = build_messages(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=many,
            intent_result=_ir(Intent.DIET),
        )
        content = msgs[1]["content"]
        # Should contain MAX_CHUNKS numbered entries, no more.
        assert f"[{MAX_CHUNKS}]" in content
        assert f"[{MAX_CHUNKS + 1}]" not in content

    def test_long_chunks_are_truncated(self, sample_blurb):
        long_text = "word " * 500
        msgs = build_messages(
            query="describe it",
            blurb=sample_blurb,
            chunks=[Chunk(text=long_text, score=0.7)],
            intent_result=_ir(Intent.DESCRIPTION),
        )
        content = msgs[1]["content"]
        # The chunk text in the prompt should be much shorter than the input.
        # We can't assert exact length because of formatting, but it should
        # be well under MAX_CHARS_PER_CHUNK + some slack.
        assert len(content) < MAX_CHARS_PER_CHUNK * 3 + MAX_BLURB_CHARS + 1000

    def test_chatml_format_has_correct_markers(self, sample_blurb, sample_chunks):
        prompt = build_prompt(
            query="what does it eat?",
            blurb=sample_blurb,
            chunks=sample_chunks,
            intent_result=_ir(Intent.DIET),
        )
        assert prompt.count("<|im_start|>") == 3  # system, user, assistant
        assert prompt.count("<|im_end|>") == 2     # only the closed turns
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_blurb_with_only_common_name_doesnt_crash(self):
        bare = Blurb(common_name="Mystery Bird")
        msgs = build_messages(
            query="what is this",
            blurb=bare,
            chunks=[],
            intent_result=_ir(Intent.DESCRIPTION),
        )
        assert "Mystery Bird" in msgs[1]["content"]

    def test_blurb_omits_empty_fields(self, sample_blurb):
        partial = Blurb(common_name="Partial Species", appearance="A thing.")
        msgs = build_messages(
            query="describe",
            blurb=partial,
            chunks=[],
            intent_result=_ir(Intent.DESCRIPTION),
        )
        content = msgs[1]["content"]
        # Should contain appearance but not the empty fields.
        assert "- appearance:" in content
        assert "- habitat:" not in content
        assert "- danger:" not in content   # no enums set, no danger line

    def test_unknown_intent_falls_back_to_other_task(self, sample_blurb):
        msgs = build_messages(
            query="anything",
            blurb=sample_blurb,
            chunks=[],
            intent_result=_ir(Intent.OTHER),
        )
        assert "TASK:" in msgs[1]["content"]

    def test_danger_intent_uses_verdict_block_not_line(self, sample_blurb):
        # Under DANGER intent, the - danger: line is replaced with a
        # VERDICT block. Both ways of expressing risk should not coexist.
        msgs = build_messages(
            query="is it safe",
            blurb=sample_blurb,
            chunks=[],
            intent_result=_ir(Intent.DANGER),
        )
        content = msgs[1]["content"]
        assert "VERDICT:" in content
        assert "- danger:" not in content

    def test_non_danger_intent_uses_danger_line(self, sample_blurb):
        # Other intents still get the compact - danger: field.
        msgs = build_messages(
            query="what does it eat",
            blurb=sample_blurb,
            chunks=[],
            intent_result=_ir(Intent.DIET),
        )
        content = msgs[1]["content"]
        assert "- danger: no (humans and pets)" in content
        assert "VERDICT:" not in content

    def test_danger_intent_no_enum_data_still_works(self):
        # Species with no risk info at all — should still produce a usable
        # prompt without crashing or claiming the species is dangerous.
        bare = Blurb(common_name="Unknown Species", appearance="something")
        msgs = build_messages(
            query="is it dangerous",
            blurb=bare,
            chunks=[],
            intent_result=_ir(Intent.DANGER),
        )
        content = msgs[1]["content"]
        assert "VERDICT:" not in content   # no verdict to render
        # The directive should tell the model to admit ignorance.
        assert "do not know" in content.lower()

    def test_invalid_danger_enum_raises(self):
        with pytest.raises(ValueError):
            Blurb(common_name="X", dangerous_to_humans="lethal")


# ===========================================================================
# Verdict + directive tests — the load-bearing logic for DANGER answers
# ===========================================================================

class TestDangerVerdict:
    """The verdict and directive together determine whether SmolLM2 says
    the right thing on safety questions. Cover the full enum matrix."""

    def test_both_no(self):
        v = _format_danger_verdict("no", "no")
        assert v is not None
        assert "NOT dangerous" in v
        # Should collapse to the combined form, not list humans/pets twice.
        assert v.count("NOT dangerous") == 1

    def test_both_yes(self):
        v = _format_danger_verdict("yes", "yes")
        assert v is not None
        assert "IS dangerous" in v

    def test_both_mild(self):
        v = _format_danger_verdict("mild", "mild")
        assert v is not None
        assert "MILD" in v

    def test_yes_humans_no_pets(self):
        v = _format_danger_verdict("yes", "no")
        assert "IS dangerous to humans" in v
        assert "is NOT dangerous to pets" in v

    def test_no_humans_yes_pets(self):
        v = _format_danger_verdict("no", "yes")
        assert "is NOT dangerous to humans" in v
        assert "IS dangerous to pets" in v

    def test_mild_humans_no_pets(self):
        v = _format_danger_verdict("mild", "no")
        assert "MILD" in v
        assert "humans" in v
        assert "NOT dangerous to pets" in v

    def test_unknown_humans_no_pets(self):
        v = _format_danger_verdict("unknown", "no")
        assert v is not None
        assert "not known" in v.lower() or "unknown" in v.lower()
        assert "NOT dangerous to pets" in v

    def test_no_humans_unknown_pets(self):
        v = _format_danger_verdict("no", "unknown")
        assert v is not None
        assert "NOT dangerous to humans" in v

    def test_both_unknown_returns_none(self):
        # No useful signal — caller should fall back.
        assert _format_danger_verdict("unknown", "unknown") is None

    def test_both_missing_returns_none(self):
        assert _format_danger_verdict(None, None) is None

    def test_one_missing_one_unknown_returns_none(self):
        # No useful signal in either field.
        assert _format_danger_verdict(None, "unknown") is None
        assert _format_danger_verdict("unknown", None) is None


class TestDangerDirective:
    """The directive tells the model how to begin its answer. Most of the
    behavior change comes from this — the verdict is the data, the
    directive is the instruction."""

    def test_safe_species_says_begin_with_no(self):
        d = _danger_directive("no", "no")
        assert '"No"' in d

    def test_dangerous_species_says_begin_with_yes(self):
        d = _danger_directive("yes", "yes")
        assert '"Yes"' in d

    def test_disagreement_does_not_pre_declare_lead(self):
        # When humans and pets disagree, the question may be specific to
        # one of them. Pre-declaring a lead word would contradict it.
        d_split = _danger_directive("no", "yes")
        d_split2 = _danger_directive("yes", "no")
        assert '"Yes"' not in d_split and '"No"' not in d_split
        assert '"Yes"' not in d_split2 and '"No"' not in d_split2
        # Should still say to use the verdict.
        assert "VERDICT" in d_split
        assert "VERDICT" in d_split2

    def test_only_humans_known_pre_declares(self):
        # If only humans is known, we can still anchor the lead word.
        d = _danger_directive("no", None)
        assert '"No"' in d
        d2 = _danger_directive("yes", None)
        assert '"Yes"' in d2

    def test_only_pets_known_pre_declares(self):
        d = _danger_directive(None, "no")
        assert '"No"' in d

    def test_mild_says_mild(self):
        d = _danger_directive("mild", "mild")
        assert "mild" in d.lower()

    def test_no_data_admits_ignorance(self):
        d = _danger_directive(None, None)
        assert "do not know" in d.lower()
        assert "caution" in d.lower()

    def test_unknown_treated_as_missing(self):
        # "unknown" in the data shouldn't make us claim safety.
        d = _danger_directive("unknown", "unknown")
        assert "do not know" in d.lower()

    def test_directive_tells_model_to_be_brief(self):
        # All directives should constrain length — verify across cases.
        for h, p in [("no", "no"), ("yes", "yes"), ("no", "yes"),
                     ("mild", "no"), (None, None), ("unknown", "unknown")]:
            d = _danger_directive(h, p)
            assert "ONE" in d or "one" in d.lower(), (
                f"directive for {h}/{p} missing length constraint: {d!r}"
            )


# ===========================================================================
# Question polarity — detecting "is it dangerous" vs "is it safe to eat"
# ===========================================================================

class TestQuestionPolarity:
    """Polarity detection routes the verdict and directive to phrasings
    that match how the visitor asked the question."""

    def test_dangerous_phrasing_is_danger(self):
        for q in [
            "is it dangerous",
            "are they dangerous to humans",
            "will it hurt me",
            "does it bite",
            "is it venomous",
            "is this poisonous",
            "is it toxic to cats",
        ]:
            assert _question_polarity(q) == "danger", q

    def test_safe_phrasing_is_safety(self):
        for q in [
            "are they safe to eat",
            "is it safe to touch",
            "are these mushrooms okay to eat",
            "is it edible",
            "ok to handle",
        ]:
            assert _question_polarity(q) == "safety", q

    def test_ambiguous_defaults_to_danger(self):
        # No keywords either way — default to the more cautious framing.
        assert _question_polarity("tell me about it") == "danger"
        assert _question_polarity("") == "danger"

    def test_both_keywords_prefers_danger(self):
        # Mixed phrasing: prefer the cautious default rather than guessing.
        assert (
            _question_polarity("is it safe or dangerous") == "danger"
        )


class TestPolarityIntegration:
    """End-to-end checks: the verdict prose and lead word match polarity."""

    def test_safety_question_dangerous_species(self):
        # Question framing is "safe?", species is dangerous.
        # Verdict should say "is NOT safe", directive should say "No".
        v = _format_danger_verdict("yes", "yes", polarity="safety")
        d = _danger_directive("yes", "yes", polarity="safety")
        assert "NOT safe" in v
        assert '"No"' in d

    def test_danger_question_dangerous_species(self):
        # "is it dangerous?" + dangerous species.
        # Verdict says "IS dangerous", directive says "Yes".
        v = _format_danger_verdict("yes", "yes", polarity="danger")
        d = _danger_directive("yes", "yes", polarity="danger")
        assert "IS dangerous" in v
        assert '"Yes"' in d

    def test_safety_question_safe_species(self):
        # "is it safe to eat?" + edible species.
        # Verdict says "IS safe", directive says "Yes".
        v = _format_danger_verdict("no", "no", polarity="safety")
        d = _danger_directive("no", "no", polarity="safety")
        assert "IS safe" in v
        assert '"Yes"' in d

    def test_danger_question_safe_species(self):
        # "is it dangerous?" + safe species. Standard case.
        v = _format_danger_verdict("no", "no", polarity="danger")
        d = _danger_directive("no", "no", polarity="danger")
        assert "NOT dangerous" in v
        assert '"No"' in d

    def test_build_messages_uses_query_polarity(self):
        # Same species, two queries with opposite polarity —
        # the rendered prompts should differ at the lead word.
        blurb = Blurb(
            common_name="Test Toxic Mushroom",
            dangerous_to_humans="yes",
            dangerous_to_pets="yes",
        )
        ir = IntentResult(Intent.DANGER, "high", 0.78, 0.10, {})

        msg_safe = build_messages(
            query="is it safe to eat",
            blurb=blurb, chunks=[], intent_result=ir,
        )[1]["content"]
        msg_danger = build_messages(
            query="is it dangerous",
            blurb=blurb, chunks=[], intent_result=ir,
        )[1]["content"]

        # Both prompts include a VERDICT line; the wording differs.
        assert "IS safe" not in msg_safe        # NOT safe, not IS safe
        assert "is NOT safe" in msg_safe
        assert "IS dangerous" in msg_danger

        # Lead words match the question framing.
        assert '"No"' in msg_safe                # "no, not safe"
        assert '"Yes"' in msg_danger             # "yes, dangerous"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])