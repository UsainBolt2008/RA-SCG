# Frozen semantic-preserving text input

The paper uses dataset-specific frozen semantic-preserving (SP)
candidate banks.

Candidate generation is not performed during RA-SCG image
optimization.

Paper-frozen SHA256 values:

RSITMD:

`5ef33aa978cb23f31d039b40835138f64e40a821f829772b86466d6b8623559f`

RSICD:

`6d92735177e6125ce5ce3479d85cb9bc8c4d0759ef360df03ee7df56ca60a366`

The offline bank-construction protocol uses BERT-MLM candidates, an
audited remote-sensing lexicon, WordNet/POS constraints, semantic and
fluency filtering, and human audit.

The frozen attack runner consumes already-selected source-specific
BA-SP captions.

It does not rebuild or search the SP bank during image optimization.
