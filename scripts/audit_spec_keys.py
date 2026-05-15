"""Audit spec-key consistency between primitive implementations and call sites.

Why this exists: every paraug primitive reads its config via
`spec.get("KEY", default)`. If a caller (test fixture, example, downstream
user) passes a dict with the wrong key name, `.get()` silently returns the
default — the call appears to work but the intended parameter is ignored.

This script does a structural diff using AST:

  1. For each primitive function in `src/paraug/{geometric,photometric}.py`,
     walk the AST body and collect every literal string argument passed to
     `spec.get(...)`. This is the "accepted" key set for that primitive.

  2. For each test file in `tests/` and each example in `examples/`, find
     every dict literal whose keys are strings AND whose values are dicts
     (the `{"primitive_name": {"p": ..., ...}}` pattern). Extract the inner
     dict's keys.

  3. Compare: every test/example key for a primitive must be in the
     primitive's accepted set, with `"p"` excluded (universal Bernoulli gate
     is read by the pipeline, not the primitive).

  4. Report mismatches; exit 1 if any are found so this can run in CI.

LIMITATION: This audit only catches **literal dict call sites** of the form
``{"primitive_name": {"key": value, ...}}``. Dynamic config construction —
e.g. ``cfg = {}; cfg["k"] = v; aug({"photometric": {"gamma": cfg}})`` —
is invisible to AST static analysis and will not be flagged. Reviewers
manually checking PRs that build specs at runtime should still grep for
``spec.get`` usage in `src/paraug/{geometric,photometric}.py` to confirm
the dynamic keys are accepted.

Run:
    python scripts/audit_spec_keys.py
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "paraug"
TESTS = ROOT / "tests"
EXAMPLES = ROOT / "examples"

UNIVERSAL_KEYS = {"p"}  # AugPipeline gate; not primitive-specific.


def primitive_accepted_keys(src_files):
    """Return {primitive_name: set(accepted_spec_keys)}."""
    accepted = {}
    for path in src_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # primitive signature: (img, mask, spec, seed_base, epoch, step) -> ...
            args = [a.arg for a in node.args.args]
            if args[:3] != ["img", "mask", "spec"]:
                continue
            keys = set()
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "get"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "spec"
                        and sub.args
                        and isinstance(sub.args[0], ast.Constant)
                        and isinstance(sub.args[0].value, str)):
                    keys.add(sub.args[0].value)
            accepted[node.name] = keys
    return accepted


def call_site_keys(file_paths, primitive_names):
    """Find {primitive_name: [(call_site_path, set(keys)), ...]} by scanning
    dict literals of the form ``{"<primitive>": {"k1": ..., "k2": ...}, ...}``.

    We require the inner value to be an ast.Dict whose keys are all constant
    strings — otherwise it could be a generic dict and a false positive.
    """
    results = defaultdict(list)
    for path in file_paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k_node, v_node in zip(node.keys, node.values):
                if not (isinstance(k_node, ast.Constant)
                        and isinstance(k_node.value, str)
                        and k_node.value in primitive_names):
                    continue
                if not isinstance(v_node, ast.Dict):
                    continue
                inner_keys = set()
                ok = True
                for ik in v_node.keys:
                    if isinstance(ik, ast.Constant) and isinstance(ik.value, str):
                        inner_keys.add(ik.value)
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                results[k_node.value].append((path.relative_to(ROOT),
                                               k_node.lineno,
                                               inner_keys))
    return results


def main():
    src_files = [SRC / "geometric.py", SRC / "photometric.py"]
    accepted = primitive_accepted_keys(src_files)

    call_site_files = (
        list(TESTS.glob("*.py"))
        + list(EXAMPLES.glob("*.py"))
    )
    primitive_names = set(accepted)
    call_sites = call_site_keys(call_site_files, primitive_names)

    mismatches = []
    for prim, sites in sorted(call_sites.items()):
        acc = accepted[prim]
        for relpath, lineno, keys in sites:
            extra = (keys - UNIVERSAL_KEYS) - acc
            if extra:
                mismatches.append((prim, relpath, lineno, sorted(extra), sorted(acc)))

    if not mismatches:
        print(f"OK — audited {len(accepted)} primitives across "
              f"{len(call_site_files)} call-site files, no spec-key mismatch.")
        return 0

    print(f"FAIL — {len(mismatches)} spec-key mismatch(es) found:")
    print()
    for prim, relpath, lineno, extra, acc in mismatches:
        print(f"  {relpath}:{lineno}  primitive {prim!r}")
        print(f"    test/example passed keys NOT in impl: {extra}")
        print(f"    impl accepts:                          {acc}")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
