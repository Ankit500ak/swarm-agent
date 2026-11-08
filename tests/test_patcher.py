import pytest

from swarm_agent.patcher import (
    PatchError,
    apply_multi_file_diff,
    apply_unified_diff,
    extract_diff,
)

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


# ---------------------------------------------------------------------------
# apply_multi_file_diff (Issue #15) - atomic, all-or-nothing, structured result
# ---------------------------------------------------------------------------

UTIL = """\
def shout(text):
    return text.lower()
"""

TWO_FILE_DIFF = """\
diff --git a/calc.py b/calc.py
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
diff --git a/util.py b/util.py
--- a/util.py
+++ b/util.py
@@ -1,2 +1,2 @@
 def shout(text):
-    return text.lower()
+    return text.upper()
"""


def _write_two_files(tmp_path):
    (tmp_path / "calc.py").write_text(ORIGINAL)
    (tmp_path / "util.py").write_text(UTIL)


def test_multi_file_success(tmp_path):
    _write_two_files(tmp_path)
    result = apply_multi_file_diff(tmp_path, TWO_FILE_DIFF)
    assert result.dry_run is False
    assert result.changed_paths == ["calc.py", "util.py"]
    by_path = {f.path: f for f in result.files}
    assert by_path["calc.py"].hunks_applied == 2
    assert by_path["calc.py"].action == "modified"
    assert by_path["util.py"].hunks_applied == 1
    assert by_path["util.py"].action == "modified"
    calc = (tmp_path / "calc.py").read_text()
    assert "return a + b" in calc
    assert "return a - b - 0" in calc
    assert "return text.upper()" in (tmp_path / "util.py").read_text()


def test_multi_file_atomicity_one_bad_file_writes_nothing(tmp_path):
    _write_two_files(tmp_path)
    bad = TWO_FILE_DIFF.replace(
        "def shout(text):", "def shout(text):  # not the real line"
    )
    with pytest.raises(PatchError):
        apply_multi_file_diff(tmp_path, bad)
    # Every file read back: the good file must NOT have been written either.
    assert (tmp_path / "calc.py").read_text() == ORIGINAL
    assert (tmp_path / "util.py").read_text() == UTIL


def test_multi_file_atomicity_blocks_creation_too(tmp_path):
    _write_two_files(tmp_path)
    diff = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
--- /dev/null
+++ b/newmod.py
@@ -0,0 +1,1 @@
+VALUE = 1
--- a/util.py
+++ b/util.py
@@ -1,2 +1,2 @@
 def whisper(text):
-    return text.lower()
+    return text.upper()
"""
    # util.py's context names a function that does not exist -> whole
    # diff must fail: calc.py unwritten, newmod.py never created.
    with pytest.raises(PatchError):
        apply_multi_file_diff(tmp_path, diff)
    assert (tmp_path / "calc.py").read_text() == ORIGINAL
    assert not (tmp_path / "newmod.py").exists()
    assert (tmp_path / "util.py").read_text() == UTIL


def test_multi_file_dry_run_writes_nothing(tmp_path):
    _write_two_files(tmp_path)
    result = apply_multi_file_diff(tmp_path, TWO_FILE_DIFF, dry_run=True)
    assert result.dry_run is True
    assert result.changed_paths == ["calc.py", "util.py"]
    assert (tmp_path / "calc.py").read_text() == ORIGINAL
    assert (tmp_path / "util.py").read_text() == UTIL


def test_multi_file_creates_new_file(tmp_path):
    (tmp_path / "calc.py").write_text(ORIGINAL)
    diff = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
--- /dev/null
+++ b/newmod.py
@@ -0,0 +1,2 @@
+def hello():
+    return "hi"
"""
    result = apply_multi_file_diff(tmp_path, diff)
    by_path = {f.path: f for f in result.files}
    assert by_path["newmod.py"].action == "created"
    assert by_path["newmod.py"].hunks_applied == 1
    assert by_path["calc.py"].action == "modified"
    assert (tmp_path / "newmod.py").read_text() == 'def hello():\n    return "hi"\n'
    assert "return a + b" in (tmp_path / "calc.py").read_text()


def test_multi_file_deletes_file(tmp_path):
    _write_two_files(tmp_path)
    diff = """\
--- a/util.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def shout(text):
-    return text.lower()
"""
    result = apply_multi_file_diff(tmp_path, diff)
    assert result.changed_paths == ["util.py"]
    assert result.files[0].action == "deleted"
    assert not (tmp_path / "util.py").exists()
    assert (tmp_path / "calc.py").read_text() == ORIGINAL


def test_unified_diff_still_handles_multi_file(tmp_path):
    # Backwards compatibility: the single entry point keeps returning a
    # plain list of paths for a multi-file diff.
    _write_two_files(tmp_path)
    changed = apply_unified_diff(tmp_path, TWO_FILE_DIFF)
    assert changed == ["calc.py", "util.py"]
    assert "return text.upper()" in (tmp_path / "util.py").read_text()
