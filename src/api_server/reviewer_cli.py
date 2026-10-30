"""Terminal-only reviewer enrollment: ``python -m src.api_server.reviewer_cli``.

Commands::

    enroll "Dr Jane Doe"   # interactive confirmation; prints the token exactly once
    list                   # enrolled reviewers (ids, names, status) — never tokens
    revoke <reviewer_id>   # interactive confirmation

``enroll`` and ``revoke`` refuse to run unless stdin and stdout are terminals and the
operator re-types the confirmation. This stops non-interactive automation (an agent
invoking the command through a tool call) from minting credentials by accident; it is
*not* a defence against a process that can drive a pseudo-terminal as the same OS user.
See ARCHITECTURE.md §6 for the full threat model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from sqlalchemy.engine import Engine

from src.core.config import get_settings
from src.state.db import init_db, make_engine, make_session_factory, prepare_sqlite_path
from src.state.repository import NotFoundError, Repository


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def main(
    argv: Sequence[str] | None = None,
    *,
    database_url: str | None = None,
    is_interactive: Callable[[], bool] = _interactive,
    prompt: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(prog="reviewer_cli", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    enroll = sub.add_parser("enroll", help="enroll a human reviewer and print a token once")
    enroll.add_argument("display_name")
    sub.add_parser("list", help="list enrolled reviewers")
    revoke = sub.add_parser("revoke", help="revoke a reviewer's token")
    revoke.add_argument("reviewer_id")
    args = parser.parse_args(argv)

    if args.command in ("enroll", "revoke") and not is_interactive():
        out("refusing to run: this command must be run by a human in an interactive terminal")
        return 2

    url = database_url or get_settings().database_url
    prepare_sqlite_path(url)
    engine = make_engine(url)
    try:
        init_db(engine)
        return _run(args, engine, prompt, out)
    finally:
        engine.dispose()


def _run(
    args: argparse.Namespace,
    engine: Engine,
    prompt: Callable[[str], str],
    out: Callable[[str], None],
) -> int:
    with make_session_factory(engine)() as session:
        repo = Repository(session)
        if args.command == "enroll":
            name = args.display_name.strip()
            if prompt(f"Re-type the reviewer name to confirm ({name}): ").strip() != name:
                out("confirmation did not match; nothing was created")
                return 1
            reviewer_id, token = repo.enroll_reviewer(name)
            session.commit()
            out(f"Enrolled reviewer {name} (id {reviewer_id}).")
            out("Token (shown once; store it privately, never in .env or the repository):")
            out(token)
            return 0
        if args.command == "list":
            for r in repo.list_reviewers():
                state = "revoked" if r.revoked_at else "active"
                out(f"{r.id}  {state:<8} {r.display_name}")
            return 0
        if prompt(f"Type REVOKE to revoke {args.reviewer_id}: ").strip() != "REVOKE":
            out("not confirmed; nothing was changed")
            return 1
        try:
            repo.revoke_reviewer(args.reviewer_id)
        except NotFoundError as exc:
            out(str(exc))
            return 1
        session.commit()
        out(f"Revoked {args.reviewer_id}.")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
