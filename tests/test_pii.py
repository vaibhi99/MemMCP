"""Tests for PII detection and redaction."""

from memmcp.privacy import PIIScanner


def test_detects_and_redacts_email():
    result = PIIScanner().scan("Reach me at jane.doe@example.com please")
    assert "email" in result.types
    assert "jane.doe@example.com" not in result.redacted
    assert "[REDACTED_EMAIL]" in result.redacted


def test_detects_api_key():
    result = PIIScanner().scan("key is sk-abcdefghijklmnopqrstuvwxyz12345")
    assert "openai_key" in result.types
    assert "[REDACTED_API_KEY]" in result.redacted


def test_valid_credit_card_flagged_via_luhn():
    result = PIIScanner().scan("card 4111 1111 1111 1111 on file")
    assert "credit_card" in result.types


def test_invalid_long_number_not_flagged_as_card():
    # Fails the Luhn check, so it should not be treated as a card.
    result = PIIScanner().scan("order number 1234567812345670000")
    assert "credit_card" not in result.types


def test_phone_number_detected():
    result = PIIScanner().scan("call 555-123-4567")
    assert "phone" in result.types


def test_clean_text_has_no_pii():
    result = PIIScanner().scan("User prefers TypeScript and dislikes ORMs")
    assert not result.has_pii
    assert result.redacted == "User prefers TypeScript and dislikes ORMs"
