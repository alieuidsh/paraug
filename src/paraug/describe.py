"""Primitive introspection helper: docstring + spec keys + defaults.

Without this you have to grep paraug source to find out what spec keys
each primitive accepts. `describe(name)` prints the docstring, then the
list of `spec.get(...)` keys with their default values pulled by AST
walk of the primitive function.

Examples:
    >>> import paraug
    >>> paraug.describe("affine")            # one primitive
    >>> paraug.describe()                    # all 31, summary
    >>> info = paraug.describe("affine", return_dict=True)
    >>> info["spec_keys"]
    {'p': 1.0, 'rot_deg': 12.0, 'scale_range': (0.9, 1.1), ...}
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Dict, Optional, Union

from .geometric import GEOMETRIC_PRIMITIVES
from .photometric import PHOTOMETRIC_PRIMITIVES


def _registry() -> Dict[str, dict]:
    """Combined primitive registry tagged by kind."""
    out = {}
    for name, fn in GEOMETRIC_PRIMITIVES.items():
        out[name] = {"kind": "geometric", "fn": fn}
    for name, fn in PHOTOMETRIC_PRIMITIVES.items():
        out[name] = {"kind": "photometric", "fn": fn}
    return out


def _ast_literal(node: ast.AST):
    """Best-effort: turn an AST node back into a Python literal. Returns the
    unparse-string for nodes that aren't pure literals."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return ast.unparse(node)


def _extract_spec_keys(fn) -> Dict[str, object]:
    """Walk the function body and collect every `spec.get(key, default)`
    call's (key, default) pair. Preserves discovery order (= source order)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return {}

    keys: Dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get"
                and isinstance(func.value, ast.Name) and func.value.id == "spec"):
            continue
        if len(node.args) < 1:
            continue
        key_node = node.args[0]
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        name = key_node.value
        if name in keys:
            continue
        default = _ast_literal(node.args[1]) if len(node.args) >= 2 else None
        keys[name] = default
    return keys


def describe(
    name: Optional[str] = None,
    *,
    return_dict: bool = False,
) -> Optional[Union[dict, Dict[str, dict]]]:
    """Print (or return) primitive introspection.

    Args:
        name: a primitive name (e.g. "affine"). If None, prints a one-line
            summary for every primitive instead.
        return_dict: if True, return the introspection instead of printing.
            With `name=None`, returns `{name: info}` for every primitive.

    The info dict has keys: ``kind`` ("geometric" | "photometric"),
    ``docstring``, ``spec_keys`` (ordered map of key→default).
    """
    reg = _registry()

    if name is None:
        summary = {
            n: {
                "kind": v["kind"],
                "docstring": (v["fn"].__doc__ or "").strip().splitlines()[0]
                              if v["fn"].__doc__ else "",
                "spec_keys": _extract_spec_keys(v["fn"]),
            }
            for n, v in reg.items()
        }
        if return_dict:
            return summary
        geo = [n for n, v in summary.items() if v["kind"] == "geometric"]
        photo = [n for n, v in summary.items() if v["kind"] == "photometric"]
        print(f"paraug primitives: {len(geo)} geometric + {len(photo)} photometric "
              f"= {len(reg)} total\n")
        print(f"Geometric ({len(geo)}):")
        for n in sorted(geo):
            print(f"  {n:<24} {summary[n]['docstring']}")
        print(f"\nPhotometric ({len(photo)}):")
        for n in sorted(photo):
            print(f"  {n:<24} {summary[n]['docstring']}")
        print("\nCall describe('<name>') for spec keys and full docstring.")
        return None

    if name not in reg:
        candidates = ", ".join(sorted(reg)[:6])
        raise KeyError(
            f"unknown primitive {name!r}. Try one of: {candidates}, … "
            f"or call describe() with no args for the full list.")

    fn = reg[name]["fn"]
    info = {
        "kind": reg[name]["kind"],
        "docstring": (fn.__doc__ or "").strip(),
        "spec_keys": _extract_spec_keys(fn),
    }
    if return_dict:
        return info

    print(f"{name} ({info['kind']})")
    print("=" * (len(name) + len(info["kind"]) + 3))
    if info["docstring"]:
        print(info["docstring"])
        print()
    print("spec keys (with defaults):")
    if info["spec_keys"]:
        for k, v in info["spec_keys"].items():
            print(f"  {k:<22} = {v!r}")
    else:
        print("  (none detected)")
    return None
