from __future__ import annotations

import torch


def conditional_asr(
    clean_ranks: torch.Tensor,
    adversarial_ranks: torch.Tensor,
    k: int,
) -> dict:
    clean_correct = (
        clean_ranks <= int(k)
    )

    attack_success = (
        clean_correct
        & (
            adversarial_ranks
            > int(k)
        )
    )

    denominator = int(
        clean_correct.sum()
    )

    successes = int(
        attack_success.sum()
    )

    return {
        "K": int(k),
        "clean_correct_denominator": denominator,
        "attack_successes": successes,
        "ASR": (
            100.0
            * successes
            / denominator
            if denominator
            else None
        ),
    }


def summarize_asr(
    clean_ranks: torch.Tensor,
    adversarial_ranks: torch.Tensor,
):
    return {
        f"ASR@{k}": conditional_asr(
            clean_ranks,
            adversarial_ranks,
            k,
        )
        for k in (
            1,
            5,
            10,
        )
    }


def macro_bb4(
    scores,
) -> float:
    """
    Macro-average the eight non-source transfer scores for one
    active source:

        four victims x two retrieval directions.
    """

    values = [
        float(x)
        for x in scores
    ]

    if len(values) != 8:
        raise ValueError(
            "BB4 requires exactly eight non-source "
            "transfer scores."
        )

    return (
        sum(values)
        / 8.0
    )
