import torch

from evaluation.fixed_gallery import (
    rank_from_scores,
)

from evaluation.metrics import (
    conditional_asr,
)


def test_best_positive_rank():
    scores = torch.tensor(
        [
            0.1,
            0.7,
            0.3,
            0.6,
        ]
    )

    rank = rank_from_scores(
        scores,
        [
            1,
            3,
        ],
    )

    assert rank == 1


def test_conditional_asr():
    clean = torch.tensor(
        [
            1,
            2,
            8,
            1,
        ]
    )

    adv = torch.tensor(
        [
            4,
            1,
            9,
            3,
        ]
    )

    out = conditional_asr(
        clean,
        adv,
        1,
    )

    assert (
        out[
            "clean_correct_denominator"
        ]
        == 2
    )

    assert (
        out[
            "attack_successes"
        ]
        == 2
    )

    assert (
        out["ASR"]
        == 100.0
    )
