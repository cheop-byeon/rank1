from __future__ import annotations
import argparse
import sys
import logging
from typing import Optional
from pathlib import Path

import mteb
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

from prompts import get_prompt
from rank1 import rank1


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = ".dataset/ids-supp/i2c/test"
RESULTS_BASE_PATH = "./results"
BATCH_SIZE = 16
BM25_BATCH_SIZE = 128
DEFAULT_ENCODE_KWARGS = {"batch_size": BATCH_SIZE}
BM25_TOP_K = 1000
RERANKER_TOP_K = 50


def _create_retrieval_task(metadata: TaskMetadata) -> AbsTaskRetrieval:
    """
    Factory function to create a custom retrieval task with dynamic metadata.
    
    This approach is more flexible than hardcoding metadata in a class definition
    and allows for runtime-provided configurations.
    
    Args:
        metadata: TaskMetadata object containing task configuration
        
    Returns:
        An instance of AbsTaskRetrieval with the provided metadata
    """
    class DynamicRetrievalTask(AbsTaskRetrieval):
        pass
    
    DynamicRetrievalTask.metadata = metadata
    return DynamicRetrievalTask()


def _get_corpus_size(dataset_path: str) -> Optional[int]:
    """
    Get the number of documents in the corpus for dynamic top_k calculation.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Number of corpus documents, or None if not found
    """
    try:
        corpus_file = Path(dataset_path) / "corpus.jsonl"
        if corpus_file.exists():
            with corpus_file.open("r", encoding="utf-8") as f:
                count = sum(1 for _ in f if _.strip())
            logger.info(f"Corpus size detected: {count} documents")
            return min(count, 1000)  # Cap at 1000
    except Exception as e:
        logger.warning(f"Failed to determine corpus size: {e}")
    return None


def _extract_model_name(model_path: str) -> str:
    """
    Extract model name from a path or model identifier.
    
    Args:
        model_path: Path to model (e.g., "/path/to/rank1-32b") or model name (e.g., "bm25s")
        
    Returns:
        Model name (last component of path, or the identifier itself)
    """
    if "/" in model_path or "\\" in model_path:
        # It's a path, extract the last component
        return Path(model_path).name
    else:
        # It's a model identifier like "bm25s"
        return model_path


def run_evaluation(dataset_name: str, subtask: Optional[str], num_gpus: int, skip_prompt: bool, model_path: str, retriever_model: str = "bm25s", split: str = "test", direction: str = "i2c") -> None:
    """Run MTEB evaluation with configurable retriever, reranker, split, and direction."""
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("MTEB Evaluation Configuration")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Split: {split}")
    logger.info(f"Direction: {direction}")
    logger.info(f"Retriever: {retriever_model}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Skip Prompt: {skip_prompt}")
    logger.info("=" * 60)
    
    # Determine prompt
    prompt = None if (skip_prompt or get_prompt(dataset_name, subtask) is None) else get_prompt(dataset_name, subtask)
    if prompt:
        logger.info(f"Using prompt: {prompt[:50]}..." if len(str(prompt)) > 50 else f"Using prompt: {prompt}")

    # Normalize subtask
    if subtask == "default":
        subtask = None

    # Create metadata and evaluation task
    metadata = TaskMetadata(
        name=dataset_name,
        description="Retrieval",
        reference=None,
        type="Retrieval",
        category="p2p",
        modalities=["text"],
        eval_splits=[split],
        eval_langs=["eng-Latn"],
        main_score="mrr_at_10",
        dataset={
            "path": f"{DATASET_PATH.replace('/test', '')}/{split}",
            "revision": "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab",
        },
        date=("2012-01-01", "2020-01-01"),
        domains=["Programming"],
        task_subtypes=["Code retrieval", "Reasoning as Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
    )
    
    task = _create_retrieval_task(metadata)
    evaluation = mteb.MTEB(tasks=[task])

    # Stage 1: BM25 retrieval
    logger.info("\n" + "=" * 60)
    logger.info(f"Stage 1: {retriever_model.upper()} Retrieval")
    logger.info("=" * 60)
    
    retriever_name = _extract_model_name(retriever_model)
    stage1_output_folder = f"{RESULTS_BASE_PATH}/stage1/{split}/{direction}/{retriever_name}"
    retriever = mteb.get_model(retriever_model)
    
    # Detect corpus size for top_k
    corpus_top_k = _get_corpus_size(DATASET_PATH)
    bm25_top_k = corpus_top_k if corpus_top_k else BM25_TOP_K
    
    try:
        evaluation.run(
            retriever,
            encode_kwargs={"batch_size": BM25_BATCH_SIZE},
            top_k=bm25_top_k,
            output_folder=stage1_output_folder,
            save_predictions=True,
            overwrite_results=False,
            co2_tracker=False,
        )
        logger.info(f"✓ {retriever_model} retrieval completed successfully")
    except Exception as e:
        logger.error(f"✗ {retriever_model} retrieval failed: {e}", exc_info=True)
        raise
    
    # Stage 2: Rank1 reranking
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: Rank1 Reranking")
    logger.info("=" * 60)
    
    try:
        reranker = rank1(
            model_name_or_path=model_path,
            num_gpus=num_gpus,
            device="cuda",
            context_size=16000,
            max_output_tokens=4096,
            fp_options="bfloat16",
        )
        logger.info(f"Model loaded: {model_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}", exc_info=True)
        raise

    # Construct previous_results path based on dataset name
    previous_results_path = f"{stage1_output_folder}/{dataset_name}_default_predictions.json"
    
    # Extract reranker model name from path
    reranker_name = _extract_model_name(model_path)
    stage2_output_folder = f"{RESULTS_BASE_PATH}/stage2/{split}/{direction}/{retriever_name}/{reranker_name}"
    
    try:
        evaluation.run(
            reranker,
            encode_kwargs=DEFAULT_ENCODE_KWARGS,
            top_k=RERANKER_TOP_K,
            output_folder=stage2_output_folder,
            save_predictions=True,
            previous_results=previous_results_path,
            overwrite_results=False,
        )
        logger.info("✓ Reranking completed successfully")
    except Exception as e:
        logger.error(f"✗ Reranking failed: {e}", exc_info=True)
        raise
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation completed successfully!")
    logger.info("=" * 60)


def main() -> int:
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Run MTEB evaluation")
    parser.add_argument("-d", "--dataset", required=True, help="MTEB dataset name")
    parser.add_argument("-n", "--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("-m", "--model_path", required=True, help="Path to the model")
    parser.add_argument("-r", "--retriever", type=str, default="bm25s", help="Retriever model (default: bm25s)")
    parser.add_argument("-s", "--split", type=str, default="test", help="Data split (default: test)", choices=["test", "dev"])
    parser.add_argument("--direction", type=str, default="i2c", help="Direction (default: i2c)", choices=["i2c", "c2i"])
    parser.add_argument("-p", "--skip_prompt", action="store_true", help="Skip prompt")
    args = parser.parse_args()

    try:
        run_evaluation(args.dataset.strip(), None, args.num_gpus, args.skip_prompt, args.model_path, args.retriever, args.split, args.direction)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

