"""Job cancellation state (``cancel.py``): which job a cancel reaches, and when."""

from __future__ import annotations

import pytest

import cancel


@pytest.fixture(autouse=True)
def _clean():
    cancel._ASKED.clear()
    yield
    cancel._ASKED.clear()


def test_running_job_stops_at_its_next_check():
    with pytest.raises(cancel.Cancelled):
        with cancel.job("a"):
            cancel.check()
            assert cancel.ask("a") is True
            assert cancel.is_cancelled()
            cancel.check()
    # Done with: the next job is not touched, and the id is forgotten.
    with cancel.job("b"):
        cancel.check()
    assert not cancel._ASKED


def test_other_jobs_cancels_do_not_reach_the_running_one():
    with cancel.job("a"):
        assert cancel.ask("b") is False
        cancel.check()


def test_job_cancelled_before_it_starts_stops_as_it_starts():
    assert cancel.ask("queued") is False
    with pytest.raises(cancel.Cancelled):
        with cancel.job("queued"):
            pytest.fail("a cancelled job must not run")
    with cancel.job("queued"):  # an id is used once; a new run under it is not cancelled
        cancel.check()


def test_no_job_no_cancel():
    cancel.ask("a")
    assert not cancel.is_cancelled()
    cancel.check()
    with cancel.job(None):
        cancel.check()


def test_remembers_a_bounded_number_of_unstarted_cancels():
    for index in range(cancel.REMEMBERED + 10):
        cancel.ask(f"job-{index}")
    assert len(cancel._ASKED) == cancel.REMEMBERED
    assert "job-0" not in cancel._ASKED
    assert f"job-{cancel.REMEMBERED + 9}" in cancel._ASKED
