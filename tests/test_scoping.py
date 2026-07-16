"""Tests for scope parsing and hierarchy."""

import pytest

from memmcp import scoping


def test_normalize_defaults_to_global():
    assert scoping.normalize(None) == "global"
    assert scoping.normalize("") == "global"
    assert scoping.normalize("GLOBAL") == "global"


def test_make_builds_hierarchical_scope():
    assert scoping.make(project="Acme", tool="Cursor") == "project:acme/tool:cursor"
    assert scoping.make(project="acme") == "project:acme"
    assert scoping.make() == "global"


def test_ancestors_widen_from_specific_to_global():
    assert scoping.ancestors("project:acme/tool:cursor") == [
        "project:acme/tool:cursor",
        "project:acme",
        "global",
    ]


def test_readable_scopes_include_self_and_ancestors():
    assert "global" in scoping.readable_scopes("project:acme")
    assert scoping.readable_scopes("global") == ["global"]


def test_invalid_scope_raises():
    with pytest.raises(ValueError):
        scoping.normalize("not-a-scope")
    with pytest.raises(ValueError):
        scoping.normalize("bogus:name")
