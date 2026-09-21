from ra_scg.core import (
    RASCGAblationAttacker,
    VIEW_NAMES,
)


def test_full_contract():
    x = RASCGAblationAttacker()

    assert x.use_relation
    assert x.use_relation_confidence
    assert x.use_directional_consensus
    assert x.use_semantic_quality

    assert tuple(
        x.active_views
    ) == tuple(
        VIEW_NAMES
    )

    assert x.steps == 10

    assert abs(
        x.epsilon - 2 / 255
    ) < 1e-12

    assert abs(
        x.step_size
        - 0.5 / 255
    ) < 1e-12

    assert abs(
        x.relation_lambda
        - 0.5
    ) < 1e-12

    assert abs(
        x.alpha - 3.0
    ) < 1e-12
