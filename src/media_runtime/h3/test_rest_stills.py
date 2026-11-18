"""imagegen ``rest_frame``: which take is retaken, which is delivered."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from h3.rest_stills import MIN_IDENTITY, MOUTH_CLOSED, Face, Still, Take, judge, keep

PERSON = Face(mouth=0.002, identity=np.array([1.0, 0.0]))


def _face(mouth: float, identity: float) -> Face:
    return Face(mouth=mouth, identity=np.array([identity, np.sqrt(1 - identity**2)]))


def _take(seed: int, face: Face | None, reference: Face | None = PERSON) -> Take:
    return judge(Take(seed=seed, path=Path(f"{seed}.png")), face, reference)


def test_takes_are_judged_on_face_identity_then_mouth():
    assert _take(1, _face(0.001, 0.8)).problem is None
    assert _take(2, _face(0.04, 0.8)).problem == "mouth"
    assert _take(3, _face(0.001, MIN_IDENTITY - 0.01)).problem == "identity"
    assert _take(4, _face(0.04, MIN_IDENTITY - 0.01)).problem == "identity"
    assert _take(5, None).problem == "no_face"
    # No face in the image: nothing to hold the still to.
    assert _take(6, None, reference=None).problem is None


def test_keep_prefers_a_clean_take_then_the_most_closed_mouth_that_kept_the_person():
    clean = _take(3, _face(0.002, 0.7))
    assert keep([_take(1, _face(0.03, 0.9)), _take(2, _face(0.001, 0.3)), clean]) is clean
    mouths = [_take(1, _face(0.05, 0.9)), _take(2, _face(MOUTH_CLOSED + 0.001, 0.6)), _take(3, _face(0.001, 0.2))]
    assert keep(mouths).seed == 2


def test_keep_delivers_the_most_similar_face_when_every_take_lost_the_person():
    assert keep([_take(1, _face(0.001, 0.2)), _take(2, None), _take(3, _face(0.03, 0.4))]).seed == 3
    assert keep([_take(1, None), _take(2, None)]).seed == 1


def test_a_still_that_missed_the_gate_reports_its_best_take_as_not_passed():
    takes = [_take(1, _face(0.001, 0.2)), _take(2, _face(0.001, 0.4))]
    still = Still(index=0, takes=takes, kept=keep(takes), render_size=(8, 8), metrics={}, checked=True)
    assert still.likeness() == {"checked": True, "takes": 2, "chosen": 1, "score": 0.4, "passed": False,
                                "problem": "identity"}
    clean = [_take(1, _face(0.001, 0.9))]
    assert Still(index=0, takes=clean, kept=keep(clean), render_size=(8, 8), metrics={},
                 checked=True).likeness()["passed"] is True
    # Nothing to hold the take to (a cartoon, or no face in the image): passed is unknown, not true.
    unchecked = [_take(1, None, reference=None)]
    assert Still(index=0, takes=unchecked, kept=keep(unchecked), render_size=(8, 8), metrics={},
                 checked=False).likeness() == {"checked": False, "takes": 1, "chosen": 0, "score": None,
                                               "passed": None, "problem": None}
