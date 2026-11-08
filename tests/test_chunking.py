from swarm_agent.chunking import chunk_file, estimate_tokens


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("a" * 100) == 25


def test_python_ast_units(tmp_path):
    src = (
        '"""Module docstring."""\n'
        "import os\n\n\n"
        "def alpha():\n    return 1\n\n\n"
        "class Beta:\n    def method(self):\n        return 2\n\n\n"
        "def gamma():\n    return 3\n"
    )
    f = tmp_path / "mod.py"
    f.write_text(src)
    chunks = chunk_file(f, budget_tokens=1500)
    kinds = [c.kind for c in chunks]
    assert "function" in kinds
    assert "class" in kinds
    texts = "\n".join(c.text for c in chunks)
    assert "def alpha" in texts
    assert "class Beta" in texts
    assert "def gamma" in texts
    # chunks start at real line numbers
    alpha = [c for c in chunks if "def alpha" in c.text][0]
    assert alpha.start_line == 5


def test_python_budget_split(tmp_path):
    body = "\n".join(f"    x{i} = {i}  # padding padding padding" for i in range(200))
    src = f"def big():\n{body}\n    return x0\n"
    f = tmp_path / "big.py"
    f.write_text(src)
    chunks = chunk_file(f, budget_tokens=50)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c.text) <= 50
    assert all(c.kind == "function" for c in chunks)


def test_python_syntax_error_falls_back(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def nope(:\n    this is not python at all :::\n")
    chunks = chunk_file(f, budget_tokens=100)
    assert chunks
    assert chunks[0].kind in ("window", "file")


def test_text_windows_with_overlap(tmp_path):
    lines = [f"line {i} with some content here" for i in range(100)]
    f = tmp_path / "notes.txt"
    f.write_text("\n".join(lines))
    chunks = chunk_file(f, budget_tokens=40)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c.text) <= 40
    # overlap: the last line of chunk N reappears in chunk N+1
    assert chunks[0].text.splitlines()[-1] in chunks[1].text


def test_small_file_single_chunk(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("hello world\n")
    chunks = chunk_file(f, budget_tokens=1500)
    assert len(chunks) == 1
    assert chunks[0].kind == "file"
    assert chunks[0].start_line == 1


def test_empty_file_no_chunks(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    assert chunk_file(f) == []
