# Rank1 Evaluation Pipeline for CodeConvo Dataset

This repository contains an optimized evaluation pipeline for using **Rank1** as a reranker in a two-stage information retrieval system. The pipeline is designed for efficient batch evaluation on the CodeConvo internet-drafts (I-Ds) dataset with support for multiple models, retrievers, splits, and directions.

**Features:**
- Two-stage retrieval pipeline (BM25/dense retrieval → Rank1 reranking)
- Configurable retrievers (bm25s, e5-small, etc.) and rerankers (rank1 models)
- Support for multiple data splits (test, dev) and directions (i2c, c2i)
- Automatic GPU allocation based on model size
- Batch SLURM job submission with parallel evaluation
- Optimized logging and result organization

## Quick Start

```bash
# Single evaluation
python test_rank1.py -d "ids-supp" -m "./models/rank1-32b" -n 2

# Batch evaluation (submit all models)
bash submit_all_evaluations.sh
```

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Citing](#citing)
- [License](#license)

## Installation 

To reproduce the experiments, we recommend using conda environment to handle dependencies:

```bash
git clone https://github.com/orionw/rank1.git
cd rank1/

conda create -p /path/to/.conda/envs/rank1 python=3.10
conda activate /path/to/.conda/envs/rank1

# Install dependencies
pip install -r requirements.txt
pip install -e mteb_branch/
pip install --no-build-isolation xformers==0.0.28.post3
pip install vllm==0.7.2
pip install flash-attn
```

## Usage

### MTEB Evaluation with test_rank1.py
For advanced evaluation with configurable retrievers, rerankers, splits, and directions:

```bash
# Basic usage with default BM25 retriever, test split, i2c direction
python test_rank1.py -d "ids-supp" -m "/path/to/rank1-32b" -n 2

# With custom retriever model
python test_rank1.py -d "ids-supp" -m "/path/to/rank1-32b" -r "bm25s" -n 2

# With custom split and direction
python test_rank1.py -d "ids-supp" -m "/path/to/rank1-32b" -s "dev" --direction "c2i" -n 2

# With all options
python test_rank1.py -d "ids-supp" -m "/path/to/rank1-32b" -r "e5-small" -s "test" --direction "i2c" -n 2 -p
```

**Arguments:**
- `-d, --dataset`: MTEB dataset name (required)
- `-m, --model_path`: Path to the reranker model (required)
- `-n, --num_gpus`: Number of GPUs to use (default: 1)
- `-r, --retriever`: Retriever model to use (default: "bm25s")
- `-s, --split`: Data split to evaluate (default: "test", choices: "test", "dev")
- `--direction`: Direction for evaluation (default: "i2c", choices: "i2c", "c2i")
- `-p, --skip_prompt`: Skip prompt augmentation

**Example workflows:**

```bash
# Evaluate rank1-32b with BM25 retriever on test split
python test_rank1.py -d "ids-supp" -m "./models/rank1-32b" -n 2

# Evaluate qwen-72b with e5-small retriever on dev split with c2i direction
python test_rank1.py -d "ids-supp" -m "./models/qwen-72b" -r "e5-small" -s "dev" --direction "c2i" -n 4

# Evaluate with multiple models and splits using a loop
for model in rank1-7b rank1-14b rank1-32b; do
    for split in test dev; do
        python test_rank1.py -d "ids-supp" -m "./models/$model" -r "bm25s" -s "$split" -n 2
    done
done
```

**Output Structure:**
Results are organized dynamically based on split, direction, retriever, and reranker:
```
results/
├── stage1/{split}/{direction}/{retriever_name}/
│   └── {dataset}_default_predictions.json
└── stage2/{split}/{direction}/{retriever_name}/{reranker_name}/
```

**Examples:**
```
results/stage1/test/i2c/bm25s/
results/stage1/dev/c2i/bm25s/
results/stage2/test/i2c/bm25s/rank1-32b/
results/stage2/dev/c2i/e5-small/qwen-72b/
```

**Batch Submission with Dynamic Configuration:**
To submit multiple models as SLURM jobs with optimal GPU allocation and configurable split/direction:

```bash
bash submit_all_evaluations.sh
```

Customize the evaluation by editing the configuration variables in the script:
```bash
# Edit these lines in submit_all_evaluations.sh
retriever_model="bm25s"  # Change retriever model
SPLIT="test"             # Change to "dev" for development split
DIRECTION="i2c"          # Change to "c2i" for corpus-to-internet direction
CONDA_ENV="/path/to/.conda/envs/rank1"
ACCOUNT="12345678"       # Your SLURM account
```

This will automatically allocate GPUs based on model size:
- 7B models: 1 GPU
- 14B models: 2 GPUs  
- 32B models: 2 GPUs
- 72B models: 4 GPUs


### MTEB Integration
Rank1 is compatible with the MTEB benchmarking framework. For more information on Rank1 and its capabilities, see the [official Rank1 repository](https://github.com/orionw/rank1) and [paper](http://arxiv.org/abs/2502.18418).

## Architecture

The evaluation pipeline uses a two-stage architecture:

1. **Stage 1: Dense/Lexical Retrieval** - Initial ranking using BM25, BM25s, or dense retrievers (e5-small, etc.)
   - Output: Top-1000 documents for reranking
   
2. **Stage 2: Rank1 Reranking** - Fine-grained relevance scoring using Rank1 models
   - Input: Top-1000 documents from Stage 1
   - Output: Top-50 final rankings

Results are saved at each stage with automatic GPU allocation and job management via SLURM.
If you use rank1 you can cite:

```bibtex
@misc{weller2025rank1testtimecomputereranking,
      title={Rank1: Test-Time Compute for Reranking in Information Retrieval}, 
      author={Orion Weller and Kathryn Ricci and Eugene Yang and Andrew Yates and Dawn Lawrie and Benjamin Van Durme},
      year={2025},
      eprint={2502.18418},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.18418}, 
}
```

## License
[MIT](LICENSE)
