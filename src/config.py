import torch

class Config:
    # --- MODEL CHANGE ---
    # switching from distilbert (66M params) to CodeBERT (125M params)
    # This model is SOTA for code/structure understanding.
    MODEL_NAME = "microsoft/codebert-base" 
    
    EMBEDDING_DIM = 768  # CodeBERT has the same dim as BERT/DistilBERT
    HIDDEN_DIM = 256     # Increased capacity for the GNN projection
    OUTPUT_DIM = 128     # Standard dimension for dense retrieval
    
    GNN_HEADS = 4        # More attention heads for the graph structure
    BATCH_SIZE = 16      # Standard batch size (lower to 8 if you run out of GPU memory)
    LEARNING_RATE = 1e-5 # Lower LR is safer for larger models
    EPOCHS = 20           # 5 epochs on 4k samples is a good start
    TEMPERATURE = 0.05   # Sharper temperature for harder negative mining
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Using the larger, high-quality JSON dataset
    DATASET_PATH = "query_triplet_dataset_dbpedia.json"
    SAVE_DIR = "checkpoints"
