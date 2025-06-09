# EpiMine: Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events
<br>Priyanka Kargupta, Yunyi Zhang, Yizhu Jiao, Siru Ouyang, Jiawei Han</a>


Official implementation for [ACL 2025](https://2025.aclweb.org/) main track paper: [Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](https://arxiv.org/abs/2408.04873).

![Framework Diagram of EpiMine](https://github.com/pkargupta/epimine/blob/main/framework.png)

State-of-the-art automatic event detection struggles with interpretability and adaptability to evolving large-scale key events---unlike episodic structures, which excel in these areas. Often overlooked, episodes represent cohesive clusters of core entities performing actions at a specific time and location; a partially ordered sequence of episodes can represent a key event. This paper introduces a novel task, episode detection, which identifies episodes within a news corpus of key event articles. Detecting episodes poses unique challenges, as they lack explicit temporal or locational markers and cannot be merged using semantic similarity alone. While large language models (LLMs) can aid with these reasoning difficulties, they suffer with long contexts typical of news corpora. To address these challenges, we introduce EpiMine, an unsupervised framework that identifies a key event's candidate episodes by leveraging natural episodic partitions in articles, estimated through shifts in discriminative term combinations. These candidate episodes are more cohesive and representative of true episodes, synergizing with LLMs to better interpret and refine them into final episodes. We apply EpiMine to our three diverse, real-world event datasets annotated at the episode level, where it achieves a 59.2% average gain across all metrics compared to baselines.

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

## Dataset
We provide all segmented articles for each key event in `episode_dataset/[theme]/[key_event]/[key_event]_segmented_raw.txt`. We also provide all episode-annotated articles in `groundtruth/[key_event]_groundtruth.txt`. All episode descriptions (used for article episode annotation) are provided in `groundtruth/key_event_episode_descriptions.xlsx`.


## 📖 Citations
Please cite the paper and star this repo if you use EpiMine and find it interesting/useful, thanks! Feel free to open an issue if you have any questions.

```bibtex
@inproceedings{
	kargupta2024unsupervised,
	title={Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events},
	author={Kargupta, Priyanka and Zhang, Yunyi and Jiao, Yizhu and Ouyang, Siru and Han, Jiawei},
	journal={arXiv preprint arXiv:2408.04873},
	year={2025}
}
```