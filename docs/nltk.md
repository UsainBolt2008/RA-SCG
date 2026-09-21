# NLTK runtime requirement

RA-SCG uses NLTK during clean-caption phrase parsing for English
part-of-speech tagging.

The attack runtime requires:

`averaged_perceptron_tagger_eng`

Install it once with:

```bash
python scripts/setup_nltk.py --download
export NLTK_DATA="$PWD/.nltk_data"
```

The original final experimental environment was:

`rsi_paper_baselines_recovered`

That environment used a Conda activation hook that set `NLTK_DATA`
to a machine-local project directory.

The public release intentionally removes that hidden side effect.

WordNet belongs to the offline semantic-preserving candidate-bank
construction protocol and is not required by the frozen-bank RA-SCG
image-optimization loop itself.
