# EpiMine: Unsupervised Episode Detection for Large-Scale News Events
<!--<br>Priyanka Kargupta, Yunyi Zhang, Yizhu Jiao, Siru Ouyang, Jiawei Han</a>-->
Given the saturation of real-time news accessible at our fingertips, processing the critical information of a key event has increasingly become a daunting challenge. Consequently, research on automatic event detection has only recently sought to address this issue by exploiting the hierarchical structure in which humans perceive realworld events: from themes (e.g.“2019 Hong Kong Protests”) down to key events (“July 1 Storming Legislative Building”), episodes (“protesters vandalized the Legislative Chamber”), and concrete atomic actions (“Protesters spray-painted slogans on the walls”). However, current methods fail to consider the episode level, despite humans neurologically encoding events within episodic structures and hence being the most intuitive event granularity to process. In this paper, we propose a novel task, episode detection, which seeks to detect episodes from a news corpus containing key event articles. An episode can describe a cohesive cluster of core entities (e.g., “protesters”, “police”) performing actions at a certain time and location. Additionally, an episode occurs as a significant component of a larger group of episodes that fall under a specific key event. In addition to serving as a more interpretable event granularity, automatically detecting episodes serves to be a challenging task given that, unlike at the key event and atomic action level, we cannot rely on having a specific time or location explicitly mentioned for each episode, and each key event article may feature only a subset of all episodes and/or describe the core entities and actions inconsistently. To address these challenges, we introduce EpiMine, an unsupervised episode detection framework that (1) automatically identifies the most salient, key-event-relevant terms and segments, (2) determines candidate episodes in an article based on natural episodic partitions estimated through shifts in discriminative term combinations, and (3) utilizes these candidate episodes and large language model-based reasoning to refine and form the final episode clusters. EpiMine is shown to often outperform all baselines through extensive experiments and case studies performed on three diverse real-world themes and thirty real-world key news event corpora.

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