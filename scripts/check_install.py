from __future__ import annotations

import os
import sys

import nltk
import open_clip
import torch

from ra_scg import (
    RASCGAttack,
)

from ra_scg.core import (
    RASCGAblationAttacker,
    SELECTED_LAYERS,
    VIEW_NAMES,
)

from ra_scg.utils.nltk_resources import (
    require_pos_tagger,
)


def main():
    print(
        "python:",
        sys.version,
    )

    print(
        "torch:",
        torch.__version__,
    )

    print(
        "open_clip:",
        getattr(
            open_clip,
            "__version__",
            "unknown",
        ),
    )

    print(
        "nltk:",
        nltk.__version__,
    )

    print(
        "NLTK_DATA:",
        os.environ.get(
            "NLTK_DATA"
        ),
    )

    require_pos_tagger()

    core = (
        RASCGAblationAttacker()
    )

    assert core.use_relation
    assert core.use_relation_confidence
    assert core.use_directional_consensus
    assert core.use_semantic_quality

    assert tuple(
        core.active_views
    ) == tuple(
        VIEW_NAMES
    )

    assert tuple(
        SELECTED_LAYERS
    ) == (
        2,
        5,
        8,
        11,
    )

    assert abs(
        core.relation_lambda
        - 0.5
    ) < 1e-12

    assert abs(
        core.epsilon
        - 2 / 255
    ) < 1e-12

    assert abs(
        core.step_size
        - 0.5 / 255
    ) < 1e-12

    assert core.steps == 10

    assert abs(
        core.alpha
        - 3.0
    ) < 1e-12

    print(
        "RA_SCG_INSTALL_CHECK=PASS"
    )


if __name__ == "__main__":
    main()
