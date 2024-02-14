# EpiMine: Unsupervised Episode Detection for Large-Scale News Events
<!--<br>Priyanka Kargupta, Yunyi Zhang, Yizhu Jiao, Siru Ouyang, Jiawei Han</a>-->

## Setup
We use `python=3.8`, `torch=1.12.1`, `cudatoolkit=12.0` (also works on 11.3), and a single NVIDIA GeForce GTX 1080. Other packages can be installed using:
```
pip install -r requirements.txt
```

In `run.py`, we define the list of `themes` (e.g., terrorism, natural_disasters, and politics) and `titles` (e.g., 2019_hk_legislative), which the former is simply used for the sake of input/output organization and the latter corresponds to the name of the input key event corpus. In order to run the following command after modifying the arguments as needed. The ground-truth episodes are defined within `run.py` as well.

```
python run.py
```
### Arguments
The following are the primary arguments for EpiMine (defined in run.py; modify as needed):

- `theme` $\rightarrow$ theme of key event
- `title` $\rightarrow$ key event to mine episodes for
- `gpu` $\rightarrow$ GPU to use; refer to nvidia-smi
- `output_dir` $\rightarrow$ default='final_output'; where to save the detected episodes.
- `lm_type` $\rightarrow$ default=`bbu`; used for computing word embeddings (`bbu` is bert-base-uncased)
- `layer` $\rightarrow$ default=12; last layer of BERT 
- `emb_dim` $\rightarrow$ default=768; Sentence and document embedding dimensions (default based on bert-base-uncased).
- `batch_size` $\rightarrow$ default=32; Batch size of episodes to segments to process (just for efficiency purposes).
- `doc_thresh` $\rightarrow$ default=0.25; Top articles to consider for candidate episode estimation.
- `vocab_min_occurrence` $\rightarrow$ default=1; Minimum frequency to be added into vocabulary.
- `eval_top` $\rightarrow$ default=5; Number of segments to consider for evaluation.
- `num` $\rightarrow$ default=5; Number of ground truth episodes for theme/key event.
- `trials` $\rightarrow$ default=10
- `api_key` $\rightarrow$ Anthropic API Key