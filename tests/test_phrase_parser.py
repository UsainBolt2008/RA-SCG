from ra_scg.phrase_relevance import (
    extract_phrase_views,
)


def test_phrase_parser():
    phrases = extract_phrase_views(
        "Two white planes are parked beside a blue building.",
        0,
    )

    kinds = {
        x.kind
        for x in phrases
    }

    assert "entity" in kinds
    assert "spatial" in kinds
