"""Tests for paraug.describe() introspection helper (v0.6.2)."""
import pytest

import paraug
from paraug import describe
from paraug.geometric import GEOMETRIC_PRIMITIVES
from paraug.photometric import PHOTOMETRIC_PRIMITIVES


def test_describe_summary_all(capsys):
    """describe() with no args prints a summary of every primitive."""
    describe()
    out = capsys.readouterr().out
    assert "paraug primitives:" in out
    assert "Geometric (" in out
    assert "Photometric (" in out
    # Every registered primitive should appear in the output.
    for name in list(GEOMETRIC_PRIMITIVES) + list(PHOTOMETRIC_PRIMITIVES):
        assert name in out, f"{name} missing from describe() summary"


def test_describe_single(capsys):
    describe("affine")
    out = capsys.readouterr().out
    assert "affine (geometric)" in out
    assert "spec keys" in out
    # Affine's docstring mentions rot_deg, so the key should be discovered.
    assert "rot_deg" in out


def test_describe_unknown_raises():
    with pytest.raises(KeyError, match="unknown primitive"):
        describe("not_a_primitive")


def test_describe_return_dict_summary():
    info = describe(return_dict=True)
    assert isinstance(info, dict)
    assert set(info.keys()) == (set(GEOMETRIC_PRIMITIVES) | set(PHOTOMETRIC_PRIMITIVES))
    affine = info["affine"]
    assert affine["kind"] == "geometric"
    assert isinstance(affine["spec_keys"], dict)


def test_describe_return_dict_single():
    info = describe("gamma", return_dict=True)
    assert info["kind"] == "photometric"
    assert "p" in info["spec_keys"]
    # gamma defines `gamma_range` via spec.get, default (0.6, 1.6)
    assert "gamma_range" in info["spec_keys"]


def test_describe_keys_are_strings():
    """Sanity: all extracted spec keys should be strings (AST literal extraction
    should never give us non-string keys for `spec.get("...", ...)` calls)."""
    info = describe(return_dict=True)
    for name, d in info.items():
        for k in d["spec_keys"]:
            assert isinstance(k, str), f"{name}: non-string spec key {k!r}"
