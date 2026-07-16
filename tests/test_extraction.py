"""Tests for the rule-based fact extractor."""

from memmcp.extraction.fact_extractor import RuleExtractor


def test_extracts_stack_choice_with_category_key():
    facts = RuleExtractor().extract("I use Postgres for the backend.")
    assert facts
    fact = facts[0]
    assert "Postgres" in fact.content
    assert fact.key == "user:database"


def test_extracts_preference():
    facts = RuleExtractor().extract("I prefer TypeScript over plain JS.")
    contents = " ".join(f.content for f in facts)
    assert "TypeScript" in contents
    assert any(f.key == "user:language" for f in facts)


def test_extracts_dislike():
    facts = RuleExtractor().extract("I hate ORMs.")
    assert any("dislikes" in f.content for f in facts)


def test_extracts_identity():
    facts = RuleExtractor().extract("My name is Vaibhav.")
    assert any(f.key == "user:name" for f in facts)
    assert any(f.importance >= 0.9 for f in facts)


def test_ignores_assistant_turns():
    convo = [
        {"role": "assistant", "content": "I use Postgres internally."},
        {"role": "user", "content": "I prefer Rust."},
    ]
    facts = RuleExtractor().extract(convo)
    contents = " ".join(f.content for f in facts)
    assert "Rust" in contents
    assert "Postgres" not in contents  # assistant statement is not memorised


def test_multiple_facts_from_one_conversation():
    convo = "I use Postgres. I prefer TypeScript. I hate ORMs."
    facts = RuleExtractor().extract(convo)
    assert len(facts) >= 3
