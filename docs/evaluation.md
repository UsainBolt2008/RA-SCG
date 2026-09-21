# Fixed-gallery evaluation and BB4

RA-SCG uses the frozen record-level fixed-gallery protocol.

For I2T, an adversarial image query is scored against the clean caption gallery, with the query image's five official positive captions represented by their frozen adversarial forms. For T2I, an adversarial caption query is scored against the clean image gallery, with its paired positive image represented by the adversarial image.

Conditional ASR@K is computed only over queries that are clean-correct at K.

For each active source, BB4 excludes the source encoder itself and macro-averages exactly four non-source victims x two directions = eight terms at each K. The dataset aggregate is the mean of the two source-specific BB4 values; the overall aggregate is the macro mean over RSITMD and RSICD.

The five evaluation encoders are:

- RemoteCLIP ViT-B/32
- RemoteCLIP RN50
- GeoRSCLIP ViT-B/32 RET-2
- OpenAI CLIP ViT-B/32
- OpenAI CLIP RN50

`configs/evaluators.yaml` records the checkpoint SHA256 values and local OpenAI CLIP source fingerprints used by the frozen protocol. RSICD uses full query-gallery score matrices with the frozen 2-D `torch.argsort` tie semantics; RSITMD retains the validated rowwise fixed-gallery path.
