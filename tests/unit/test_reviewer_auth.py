"""Reviewer credentials, enrollment CLI and persistence-level enforcement (PR #1 review)."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api_server.reviewer_cli import main as cli
from src.core.enums import ActorType, GateId, GateVerdict, Phase
from src.core.exceptions import SysRevError
from src.core.schemas import GateDecision
from src.core.security import format_token, hash_secret, new_salt, parse_token, verify_secret
from src.state.db import ReviewerORM
from src.state.repository import NotFoundError, Repository
from tests.conftest import READY


@pytest.fixture
def repo(session: Session) -> Repository:
    r = Repository(session)
    r.create_project("p1", "Synthetic review")
    return r


# --------------------------------------------------------------------------- primitives


def test_token_round_trip_and_hashing() -> None:
    rid = "a" * 32
    secret = "S" * 43
    assert parse_token(format_token(rid, secret)) == (rid, secret)
    for bad in ("", "rvw_short_x", f"xyz_{rid}_{secret}", f"rvw_{rid}_short"):
        assert parse_token(bad) is None
    salt = new_salt()
    digest = hash_secret(secret, salt)
    assert secret not in digest
    assert verify_secret(secret, salt, digest)
    assert not verify_secret(secret + "x", salt, digest)
    assert hash_secret(secret, new_salt()) != digest  # salted


def test_database_stores_only_a_hash(repo: Repository, session: Session) -> None:
    reviewer_id, token = repo.enroll_reviewer("Dr R")
    _, secret = parse_token(token)  # type: ignore[misc]
    row = session.scalars(select(ReviewerORM)).one()
    assert row.id == reviewer_id
    assert secret not in (row.token_hash, row.salt)
    assert repo.authenticate_reviewer(token) is row
    with pytest.raises(ValueError, match="empty"):
        repo.enroll_reviewer("   ")


def test_revocation(repo: Repository) -> None:
    reviewer_id, token = repo.enroll_reviewer("Dr R")
    repo.revoke_reviewer(reviewer_id)
    repo.revoke_reviewer(reviewer_id)  # idempotent
    assert repo.authenticate_reviewer(token) is None
    with pytest.raises(NotFoundError):
        repo.revoke_reviewer("missing")


# --------------------------------------------------------------------------- persistence


def _gate1_transition(repo: Repository, decision: GateDecision) -> None:
    m = repo.load_state_machine("p1")
    repo.record_transition(
        m.transition(Phase.GATE1_SEARCH_STRATEGY, actor="o", actor_type=ActorType.SYSTEM,
                     context=READY)
    )  # fmt: skip
    repo.record_transition(m.submit_gate_decision(decision, READY))


def _decision(**kw: object) -> GateDecision:
    data: dict[str, object] = {"gate": GateId.GATE_1, "verdict": GateVerdict.APPROVED,
                               "reviewer_name": "Dr R"}  # fmt: skip
    data.update(kw)
    return GateDecision(**data)  # type: ignore[arg-type]


def test_unauthenticated_gate_decision_cannot_be_persisted(repo: Repository) -> None:
    """A library caller claiming to be human without an enrolled reviewer id is refused."""
    with pytest.raises(SysRevError, match="enrolled reviewer"):
        _gate1_transition(repo, _decision())
    with pytest.raises(SysRevError, match="enrolled reviewer"):
        repo.save_gate_decision("p1", _decision())


def test_unknown_reviewer_cannot_be_persisted(repo: Repository) -> None:
    with pytest.raises(SysRevError, match="not enrolled"):
        _gate1_transition(repo, _decision(reviewer_id="f" * 32))


def test_revoked_or_renamed_reviewer_is_refused(session: Session) -> None:
    repo = Repository(session)
    repo.create_project("p1", "x")
    reviewer_id, _ = repo.enroll_reviewer("Dr R")
    with pytest.raises(SysRevError, match="does not match"):
        _gate1_transition(repo, _decision(reviewer_id=reviewer_id, reviewer_name="Dr Impostor"))


def test_revoked_reviewer_cannot_persist(session: Session) -> None:
    repo = Repository(session)
    repo.create_project("p1", "x")
    reviewer_id, _ = repo.enroll_reviewer("Dr R")
    repo.revoke_reviewer(reviewer_id)
    with pytest.raises(SysRevError, match="revoked"):
        _gate1_transition(repo, _decision(reviewer_id=reviewer_id))


def test_enrolled_reviewer_decision_is_persisted(session: Session) -> None:
    repo = Repository(session)
    repo.create_project("p1", "x")
    reviewer_id, _ = repo.enroll_reviewer("Dr R")
    _gate1_transition(repo, _decision(reviewer_id=reviewer_id))
    last = repo.list_transitions("p1")[-1]
    assert last.gate_decision is not None
    assert last.gate_decision.reviewer_id == reviewer_id


# --------------------------------------------------------------------------- CLI


@pytest.fixture
def db_url(tmp_path: object) -> str:
    return f"sqlite:///{tmp_path}/cli.db"


def run(db_url: str, argv: list[str], *, tty: bool = True, answers: list[str] | None = None,
        ) -> tuple[int, list[str]]:  # fmt: skip
    out: list[str] = []
    replies = iter(answers or [])
    code = cli(argv, database_url=db_url, is_interactive=lambda: tty,
               prompt=lambda _msg: next(replies), out=out.append)  # fmt: skip
    return code, out


def test_cli_refuses_non_interactive_enrollment_and_revocation(db_url: str) -> None:
    for argv in (["enroll", "Dr R"], ["revoke", "x"]):
        code, out = run(db_url, argv, tty=False)
        assert code == 2
        assert "interactive terminal" in out[0]


def test_cli_enroll_requires_confirmation(db_url: str) -> None:
    code, out = run(db_url, ["enroll", "Dr R"], answers=["Dr Wrong"])
    assert code == 1
    assert "nothing was created" in out[0]
    assert run(db_url, ["list"])[1] == []


def test_cli_enroll_list_revoke(db_url: str) -> None:
    from src.state.db import init_db, make_engine, make_session_factory

    code, out = run(db_url, ["enroll", "Dr R"], answers=["Dr R"])
    assert code == 0
    token = out[-1]
    assert parse_token(token) is not None
    code, listing = run(db_url, ["list"])
    assert code == 0
    assert "Dr R" in listing[0] and token not in listing[0]
    reviewer_id = listing[0].split()[0]

    engine = make_engine(db_url)
    init_db(engine)
    with make_session_factory(engine)() as s:
        assert Repository(s).authenticate_reviewer(token) is not None

    assert run(db_url, ["revoke", reviewer_id], answers=["no"])[0] == 1
    assert run(db_url, ["revoke", "missing"], answers=["REVOKE"])[0] == 1
    assert run(db_url, ["revoke", reviewer_id], answers=["REVOKE"])[0] == 0
    with make_session_factory(engine)() as s:
        assert Repository(s).authenticate_reviewer(token) is None
    assert "revoked" in run(db_url, ["list"])[1][0]
    engine.dispose()


def test_sqlite_directory_created_owner_only(tmp_path: object) -> None:
    import stat
    from pathlib import Path

    from src.state.db import prepare_sqlite_path

    target = Path(str(tmp_path)) / "nested" / "db"
    prepare_sqlite_path(f"sqlite:///{target}/sysrev.db")
    if os.name == "nt":  # Session 10: POSIX modes do not exist on Windows; the DACL does
        from src.state.db import directory_protection_problems

        # structured: owner and DACL by SID (src.state.acl), not icacls text or entry count
        assert directory_protection_problems(target) == []
    else:
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
    prepare_sqlite_path("sqlite:///:memory:")  # no-op
