import pytest

from swarm_agent.patcher import PatchError, apply_unified_diff, extract_diff

ORIGINAL = """\
def add(a, b):
    return a - b


def sub(a, b):
    return a - b
"""

GOOD_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def test_extract_diff_fenced():
    text = "Here is the fix:\n```diff\n" + GOOD_DIFF + "```\nDone."
    assert extract_diff(text) == GOOD_DIFF.strip("\n")


def test_extract_diff_bare():
    assert extract_diff(GOOD_DIFF) == GOOD_DIFF.strip("\n")


def test_extract_diff_none():
    assert extract_diff("no diff here, just prose") is None
    assert extract_diff("") is None


def test_apply_real_diff(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text(ORIGINAL)
    changed = apply_unified_diff(tmp_path, GOOD_DIFF)
    assert changed == ["calc.py"]
    content = f.read_text()
    assert "return a + b" in content
    # the untouched second function is preserved verbatim
    assert "def sub(a, b):\n    return a - b" in content


def test_apply_context_mismatch_changes_nothing(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text(ORIGINAL)
    bad = GOOD_DIFF.replace("def add(a, b):", "def add(a, b):  # different")
    with pytest.raises(PatchError):
        apply_unified_diff(tmp_path, bad)
    assert f.read_text() == ORIGINAL


def test_apply_dry_run_writes_nothing(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text(ORIGINAL)
    changed = apply_unified_diff(tmp_path, GOOD_DIFF, dry_run=True)
    assert changed == ["calc.py"]
    assert f.read_text() == ORIGINAL


def test_apply_creates_new_file(tmp_path):
    diff = """\
--- /dev/null
+++ b/newmod.py
@@ -0,0 +1,2 @@
+def hello():
+    return "hi"
"""
    changed = apply_unified_diff(tmp_path, diff)
    assert changed == ["newmod.py"]
    assert (tmp_path / "newmod.py").read_text() == 'def hello():\n    return "hi"\n'


def test_apply_multi_hunk(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text(ORIGINAL)
    diff = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
@@ -5,2 +5,2 @@
 def sub(a, b):
-    return a - b
+    return a - b - 0
"""
    apply_unified_diff(tmp_path, diff)
    content = f.read_text()
    assert "return a + b" in content
    assert "return a - b - 0" in content


def test_apply_path_escape_rejected(tmp_path):
    diff = """\
--- a/../evil.py
+++ b/../evil.py
@@ -1,1 +1,1 @@
-x
+y
"""
    with pytest.raises(PatchError):
        apply_unified_diff(tmp_path, diff)


def test_malformed_diff_rejected(tmp_path):
    with pytest.raises(PatchError):
        apply_unified_diff(tmp_path, "this is not a diff")
