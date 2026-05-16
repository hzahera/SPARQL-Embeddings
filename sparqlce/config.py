"""config.py
Central configuration for the HybridSPARQLEncoder training pipeline.
Edit this file to change model, training, and path settings.
"""
import torch


class Config:
    MODEL_NAME = "microsoft/codebert-base"

    EMBEDDING_DIM = 768   # CodeBERT output dimension
    HIDDEN_DIM    = 256   # GATv2 hidden dimension
    OUTPUT_DIM    = 128   # Final L2-normalised embedding dimension

    GNN_HEADS     = 4     # Number of attention heads in GATv2
    BATCH_SIZE    = 4     # Keep small to avoid OOM with full pos/neg sets
    LEARNING_RATE = 1e-5  # Conservative LR for fine-tuning CodeBERT jointly
    EPOCHS        = 20    # Training epochs; reduce to 5 for a quick smoke-test
    TEMPERATURE   = 0.05  # InfoNCE temperature

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset paths (relative to project root)
    DATASET_PATH_DBPEDIA  = "dataset/raw/jaccard_triplets_dbpedia.json"
    DATASET_PATH_WIKIDATA = "dataset/raw/jaccard_triplets_wikidata.json.gz"
    SPLITS_DIR            = "dataset/splits"
    CHECKPOINT_DIR        = "checkpoints"
