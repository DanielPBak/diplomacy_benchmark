"""Scorers against hand-computed cases."""
import json

import pytest

from dipbench.agents import RandomAgent
from dipbench.config import POWERS
from dipbench.loop import GameConfig, GameResult, run_game
from dipbench.scoring import SYSTEMS, dss, linear, linear_norm, score_all, sos

A, E, F, G, I, R, T = POWERS


def make_result(outcome, centers, winners, survivors=None, final_year=1910, max_year=1910,
                year_survivors=None):
    centers = {p: centers.get(p, 0) for p in POWERS}
    survivors = survivors if survivors is not None else [p for p in POWERS if centers[p] > 0]
    if year_survivors is None:
        year_survivors = {y: survivors for y in range(1901, final_year + 1)}
    return GameResult(game_id="t", press_mode="gunboat", outcome=outcome, winners=winners,
                      final_centers=centers, final_phase=f"W{final_year}A", final_year=final_year,
                      survivors=survivors, max_year=max_year, year_survivors=year_survivors)


def zeros_except(**scores):
    return {p: scores.get(p, 0.0) for p in POWERS}


# --- solo -----------------------------------------------------------------------------

@pytest.fixture
def solo():
    # France solos in 1907. Austria, Russia and Turkey were eliminated in 1904.
    centers = {F: 18, E: 8, G: 5, I: 3}
    ys = {y: list(POWERS) if y <= 1903 else [E, F, G, I] for y in range(1901, 1908)}
    return make_result("solo", centers, [F], survivors=[E, F, G, I], final_year=1907, year_survivors=ys)


def test_solo_dss_sos(solo):
    assert dss(solo) == zeros_except(FRANCE=1.0)
    assert sos(solo) == zeros_except(FRANCE=1.0)


def test_solo_is_full_credit_in_every_normalized_system(solo):
    for scorer in (dss, sos, linear, linear_norm):
        assert scorer(solo) == zeros_except(FRANCE=1.0)


# --- 2-way and 3-way draws ------------------------------------------------------------

def test_two_way_draw():
    r = make_result("draw_vote", {F: 17, E: 17}, [E, F], final_year=1912, max_year=1920)
    assert dss(r) == zeros_except(ENGLAND=0.5, FRANCE=0.5)
    assert sos(r) == zeros_except(ENGLAND=0.5, FRANCE=0.5)  # 289 / 578
    assert linear(r) == zeros_except(ENGLAND=0.5, FRANCE=0.5)


def test_unequal_two_way_draw_sos():
    r = make_result("draw_vote", {F: 20, E: 14}, [E, F])  # 20 centers but drawn, not soloed
    assert sos(r)[F] == pytest.approx(400 / 596)
    assert sos(r)[E] == pytest.approx(196 / 596)
    assert dss(r)[F] == 0.5


def test_three_way_draw():
    r = make_result("draw_vote", {F: 16, E: 12, G: 6}, [E, F, G])
    assert dss(r) == zeros_except(ENGLAND=1 / 3, FRANCE=1 / 3, GERMANY=1 / 3)
    # squares: 256 + 144 + 36 = 436
    assert sos(r)[F] == pytest.approx(256 / 436)
    assert sos(r)[E] == pytest.approx(144 / 436)
    assert sos(r)[G] == pytest.approx(36 / 436)
    assert linear_norm(r)[G] == pytest.approx(6 / 34)


# --- year-cap draw --------------------------------------------------------------------

def test_year_cap_draw_scores_like_a_draw():
    centers = {A: 4, E: 5, F: 6, G: 5, I: 4, R: 6, T: 4}  # = 34, all 7 survive
    cap = make_result("year_cap", centers, list(POWERS))
    vote = make_result("draw_vote", centers, list(POWERS))
    assert score_all(cap) == score_all(vote)
    assert dss(cap) == {p: 1 / 7 for p in POWERS}
    # squares: 16 + 25 + 36 + 25 + 16 + 36 + 16 = 170
    assert sos(cap)[F] == pytest.approx(36 / 170)
    assert sos(cap)[A] == pytest.approx(16 / 170)


# --- eliminations ---------------------------------------------------------------------

def test_eliminated_powers_score_zero_in_draws():
    # Italy survives with units but no centers: a draw participant with 0 centers.
    r = make_result("draw_vote", {F: 15, E: 10, G: 9}, [E, F, G, I], survivors=[E, F, G, I])
    assert dss(r) == zeros_except(ENGLAND=0.25, FRANCE=0.25, GERMANY=0.25, ITALY=0.25)
    assert sos(r)[I] == 0.0
    assert sos(r)[F] == pytest.approx(225 / 406)
    assert linear_norm(r)[I] == 0.0
    for p in (A, R, T):
        assert dss(r)[p] == sos(r)[p] == linear(r)[p] == linear_norm(r)[p] == 0.0


def test_zero_sum_systems_sum_to_one():
    for r in (make_result("draw_vote", {F: 16, E: 12, G: 6}, [E, F, G]),
              make_result("solo", {F: 19, E: 15}, [F], final_year=1906)):
        assert sum(dss(r).values()) == pytest.approx(1)
        assert sum(sos(r).values()) == pytest.approx(1)
        assert sum(linear_norm(r).values()) == pytest.approx(1)


def test_bad_outcome_rejected():
    with pytest.raises(ValueError):
        dss(make_result("timeout", {F: 34}, [F]))
    with pytest.raises(ValueError):
        dss(make_result("solo", {F: 17, E: 17}, [E, F]))


def test_scores_from_saved_result_json(tmp_path):
    result = run_game(GameConfig(press_mode="gunboat", max_year=1903), {p: RandomAgent(p, seed=3) for p in POWERS},
                      out_dir=tmp_path)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert score_all(saved) == score_all(result)
    assert sum(score_all(saved)["dss"].values()) == pytest.approx(1)
    assert set(score_all(saved)) == set(SYSTEMS)
