"""The ``swarm-agent`` command line.

Subcommands:

- ``fix``        - diagnose an error (piped on stdin, or ``--file``) against
                   the repo, propose a diff, optionally ``--apply`` it and
                   run the tests.
- ``analyze``    - fan a repository analysis out across the swarm.
- ``gen-tests``  - generate pytest tests for a module.
- ``swarm``      - show the peers that make up your swarm.
- ``ask``        - one quick question to the fastest peer.
- ``doctor``     - hardware report, backend recommendations, peer health.
- ``sessions``   - Level 2 orchestrator: run tasks as parallel sessions,
                   each in its own git worktree, optionally pinned to a peer.

Exit codes: 0 success - 1 the swarm or the workflow failed - 2 bad usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, agent, backends, orchestrator, prompts
from .config import Config, load_config
from .discovery import discover
from .ollama import SwarmError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-agent",
        description="A terminal-native coding agent that runs on the Ollama "
                    "swarm on your local network. Your code never leaves the LAN.",
    )
    parser.add_argument("--version", action="version", version=f"swarm-agent {__version__}")
    parser.add_argument("--model", help="model to request from peers (overrides config/env)")
    parser.add_argument("--peers", help="comma-separated peer base URLs (overrides config/env)")
    sub = parser.add_subparsers(dest="command")

    p_fix = sub.add_parser("fix", help="diagnose an error and propose/apply a fix")
    p_fix.add_argument("--file", help="read the error from this file instead of stdin")
    p_fix.add_argument("--path", default=".", help="repository root (default: cwd)")
    p_fix.add_argument("--apply", action="store_true",
                       help="apply the diff and run the tests (default: dry run)")
    p_fix.add_argument("--test-cmd", default=None,
                       help="test command to run after applying (overrides config)")

    p_an = sub.add_parser("analyze", help="analyse a repository across the swarm")
    p_an.add_argument("path", nargs="?", default=".", help="path to analyse (default: cwd)")
    p_an.add_argument("--question", help="a specific question to answer about the code")

    p_gt = sub.add_parser("gen-tests", help="generate pytest tests for a module")
    p_gt.add_argument("target", help="the Python module to test")
    p_gt.add_argument("--write", action="store_true",
                      help="write tests/test_<name>_generated.py instead of printing")
    p_gt.add_argument("--run", action="store_true",
                      help="run the generated tests after writing them")

    p_sw = sub.add_parser("swarm", help="list the peers in your swarm")
    p_sw.add_argument("--scan", action="store_true",
                      help="also scan the local /24 subnet for Ollama peers")

    p_ask = sub.add_parser("ask", help="ask the fastest peer a single question")
    p_ask.add_argument("prompt", nargs="+", help="the question")

    sub.add_parser("doctor", help="hardware report + peer health")

    p_sess = sub.add_parser(
        "sessions", help="Level 2 orchestrator: parallel task sessions")
    sess_sub = p_sess.add_subparsers(dest="sessions_command")

    p_spawn = sess_sub.add_parser("spawn", help="queue a session in its own worktree")
    p_spawn.add_argument("--repo", default=".", help="repository root (default: cwd)")
    p_spawn.add_argument("--kind", required=True,
                         choices=list(orchestrator.SESSION_KINDS),
                         help="which flow the session runs")
    p_spawn.add_argument("--peer", help="pin the session to one peer base URL")
    p_spawn.add_argument("--prompt", help="prompt text (kind: ask)")
    p_spawn.add_argument("--question", help="question about the code (kind: analyze)")
    p_spawn.add_argument("--target", help="module path, relative to the repo (kind: gen-tests)")
    p_spawn.add_argument("--file", help="read the error log from this file (kind: fix)")
    p_spawn.add_argument("--apply", action="store_true",
                         help="apply the fix instead of a dry run (kind: fix)")

    p_run = sess_sub.add_parser("run", help="run one queued session, or all of them")
    p_run.add_argument("id", nargs="?", help="session id (default: run all queued)")
    p_run.add_argument("--repo", default=".", help="repository root (default: cwd)")
    p_run.add_argument("--workers", type=int, default=4,
                       help="parallel sessions when running all (default: 4)")

    p_list = sess_sub.add_parser("list", help="list sessions for a repository")
    p_list.add_argument("--repo", default=".", help="repository root (default: cwd)")

    p_stop = sess_sub.add_parser("stop", help="stop a queued or running session")
    p_stop.add_argument("id", help="session id")
    p_stop.add_argument("--repo", default=".", help="repository root (default: cwd)")

    p_show = sess_sub.add_parser("show", help="show one session in full")
    p_show.add_argument("id", help="session id")
    p_show.add_argument("--repo", default=".", help="repository root (default: cwd)")

    p_clean = sess_sub.add_parser("clean", help="remove a finished session's worktree")
    p_clean.add_argument("id", help="session id")
    p_clean.add_argument("--repo", default=".", help="repository root (default: cwd)")
    p_clean.add_argument("--delete-branch", action="store_true",
                         help="also delete the session's swarm/<id> branch")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    config = load_config()
    if getattr(args, "model", None):
        config.model = args.model
    if getattr(args, "peers", None):
        from .config import _split_peers

        config.peers = _split_peers(args.peers)
    return config


def _read_error(args: argparse.Namespace) -> str | None:
    if args.file:
        try:
            return Path(args.file).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"swarm-agent: cannot read --file {args.file}: {exc}", file=sys.stderr)
            return None
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    print(
        "swarm-agent: no error text given - pipe one in "
        "(e.g. `cat error.log | swarm-agent fix`) or pass --file",
        file=sys.stderr,
    )
    return None


def _cmd_swarm(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.scan:
        config.auto_discover = True
    peers = discover(config)
    if not peers:
        print(agent.NO_PEER_HINT)
        return 1
    print(f"{'PEER':<34} {'KIND':<7} {'LATENCY':>9}  MODELS")
    for peer in peers:
        models = ", ".join(peer.models) if peer.models else "(none pulled)"
        print(f"{peer.base_url:<34} {peer.kind:<7} {peer.latency_ms:>7.0f} ms  {models}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    report = backends.hardware_report()
    print("Hardware")
    print(f"  platform       : {report['platform']}")
    print(f"  cpu cores      : {report['cpu_count']}")
    if report["memory_gb"] is not None:
        print(f"  memory         : {report['memory_gb']} GB")
    if report["apple_silicon"]:
        print(f"  chip           : {report['chip'] or 'Apple Silicon'}")
        print(f"  unified memory : {report['unified_memory_gb']} GB")
    print(f"  mlx available  : {'yes' if backends.mlx_available() else 'no'}")
    print("\nRecommendations")
    for tip in backends.recommendations(report):
        print(f"  - {tip}")
    print("\nPeers")
    config = _config_from_args(args)
    peers = discover(config)
    if not peers:
        print(f"  {agent.NO_PEER_HINT}")
        return 1
    for peer in peers:
        state = "ok" if peer.models else "up, no models pulled"
        print(f"  {peer.base_url} [{peer.kind}] {peer.latency_ms:.0f} ms - {state}")
    return 0


def _session_payload(args: argparse.Namespace) -> dict:
    if args.kind == "ask":
        return {"prompt": args.prompt or ""}
    if args.kind == "analyze":
        return {"question": args.question}
    if args.kind == "gen-tests":
        return {"target": args.target or ""}
    if args.kind == "fix":
        if not args.file:
            raise SwarmError("sessions spawn --kind fix needs --file <error log>")
        try:
            error_text = Path(args.file).read_text(
                encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SwarmError(f"cannot read --file {args.file}: {exc}") from exc
        return {"error": error_text, "apply": bool(args.apply)}
    return {}


def _print_session(session) -> None:
    peer = session.peer or "(swarm)"
    print(f"{session.id}  {session.status:<8} {session.kind:<10} {peer:<28} "
          f"{session.worktree}")


def _cmd_sessions(args: argparse.Namespace) -> int:
    if not getattr(args, "sessions_command", None):
        print("usage: swarm-agent sessions {spawn|run|list|stop|show|clean}",
              file=sys.stderr)
        return 2
    config = _config_from_args(args)
    orch = orchestrator.Orchestrator(args.repo, config)
    if args.sessions_command == "spawn":
        session = orch.spawn(
            args.kind, _session_payload(args), peer=args.peer)
        _print_session(session)
        return 0
    if args.sessions_command == "run":
        if args.id:
            finished = [orch.run(args.id)]
        else:
            finished = orch.run_all(max_workers=args.workers)
        for session in finished:
            _print_session(session)
            if session.status == "failed":
                print(f"  error: {session.error}", file=sys.stderr)
        if not finished:
            print("no queued sessions")
        return 1 if any(s.status == "failed" for s in finished) else 0
    if args.sessions_command == "list":
        sessions = orch.list()
        if not sessions:
            print("no sessions")
            return 0
        for session in sessions:
            _print_session(session)
        return 0
    if args.sessions_command == "stop":
        _print_session(orch.stop(args.id))
        return 0
    if args.sessions_command == "show":
        session = orch.get(args.id)
        print(f"id       : {session.id}")
        print(f"kind     : {session.kind}")
        print(f"status   : {session.status}")
        print(f"peer     : {session.peer or '(swarm)'}")
        print(f"branch   : {session.branch}")
        print(f"worktree : {session.worktree}")
        print(f"created  : {session.created_at}")
        print(f"started  : {session.started_at or '-'}")
        print(f"finished : {session.finished_at or '-'}")
        print(f"log      : {orch.log_path(session)}")
        if session.error:
            print(f"error    : {session.error}")
        if session.result:
            print("\n--- result ---")
            print(session.result)
        return 0
    if args.sessions_command == "clean":
        _print_session(orch.clean(args.id, delete_branch=args.delete_branch))
        return 0
    return 2


def _cmd_ask(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    distributor = agent.get_distributor(config)
    reply = distributor.ask([
        {"role": "system", "content": prompts.ASK_SYSTEM},
        {"role": "user", "content": " ".join(args.prompt)},
    ])
    print(reply)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        if args.command == "fix":
            error_text = _read_error(args)
            if error_text is None:
                return 2
            config = _config_from_args(args)
            return agent.fix_flow(
                error_text,
                repo_root=args.path,
                config=config,
                apply=args.apply,
                test_cmd=args.test_cmd,
            )
        if args.command == "analyze":
            config = _config_from_args(args)
            agent.analyze_flow(args.path, config, question=args.question)
            return 0
        if args.command == "gen-tests":
            config = _config_from_args(args)
            agent.gen_tests_flow(args.target, config, write=args.write, run=args.run)
            return 0
        if args.command == "swarm":
            return _cmd_swarm(args)
        if args.command == "ask":
            return _cmd_ask(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "sessions":
            return _cmd_sessions(args)
    except SwarmError as exc:
        print(f"swarm-agent: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
