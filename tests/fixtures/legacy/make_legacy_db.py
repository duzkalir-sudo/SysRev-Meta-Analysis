"""Create the cross-version fixture database with the *old* source (Session 9, P1a).

Run from a checkout of commit df1a79a (the last ``main`` before Session 9)::

    git worktree add /tmp/old_src df1a79a
    cd /tmp/old_src && python -I /path/to/this/make_legacy_db.py OUT.sqlite

It builds, with that commit's own code and test helpers, three SYNTHETIC projects:

* ``ms1`` - a single-arm review (GLMM proportions, Kaplan-Meier at 5 years) in the audit
  phase with a passing audit, every human-review finding acknowledged and a human sign-off;
* ``mc1`` - a comparative review (Mantel-Haenszel RR) in Stage 6 with a verified
  ``stats_output.json``;

and records the facts the new source must reproduce in ``legacy_facts`` (JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from sqlalchemy import text

from src.manuscript.service import completion_problems
from src.state.db import init_db, make_engine, make_session_factory
from tests.manuscript_helpers import comparative_project
from tests.unit.test_manuscript_review_fixes import _attested, _sign_off


def main(out: Path) -> None:
    out.unlink(missing_ok=True)
    engine = make_engine(f"sqlite:///{out}")
    init_db(engine)
    facts: dict[str, object] = {}
    with make_session_factory(engine)() as s:
        svc, _ = _attested(s)
        _sign_off(svc, _)
        repo = svc.repo
        assert repo.stats_output_problems("ms1") == []
        assert completion_problems(repo, "ms1") == []
        repo2, _, _ = comparative_project(s)
        assert repo2.stats_output_problems("mc1") == []
        for pid in ("ms1", "mc1"):
            project = repo.get_project(pid)
            facts[pid] = {
                "phase": project.phase,
                "stats_output_sha256": project.stats_output_sha256,
                "input_sha256": repo.load_stage5_inputs(pid).sha256(),
            }
        s.commit()
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE legacy_facts (payload TEXT NOT NULL)"))
        conn.execute(text("INSERT INTO legacy_facts VALUES (:p)"), {"p": json.dumps(facts)})
    engine.dispose()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
