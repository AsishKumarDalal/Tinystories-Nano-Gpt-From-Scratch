# TinyStories Nano-GPT 🤖📚

A lightweight GPT implementation trained on the TinyStories dataset, demonstrating fundamental transformer architecture and language modeling capabilities.

## Overview

This project implements a small-scale GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. The model is trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), a collection of short stories designed for training and evaluating language models on narrative text generation.

### Key Features

- *Custom Transformer Architecture*: Built from scratch with multi-head attention, feed-forward networks, and layer normalization
- *Efficient Training*: Optimized data loading with memory-mapped binary files for fast batch processing
- *Text Generation*: Supports temperature-based sampling and top-k filtering for controlled generation
- *Modular Design*: Clean, educational code structure suitable for learning transformer implementations

## Model Architecture

### Configuration (49M Parameter Variant)

python
vocab_size: 50,257 tokens (GPT-2 tokenizer)
context_length: 128 tokens
embedding_dim: 384
attention_heads: 6
transformer_layers: 6
dropout_rate: 0.1


### Components

1. *Token & Position Embeddings*: Learned representations for tokens and their positions
2. *Multi-Head Attention*: Parallel attention mechanisms with causal masking for autoregressive generation
3. *Feed-Forward Networks*: Position-wise MLP with GELU activation (4x expansion ratio)
4. *Layer Normalization*: Pre-normalization for stable training
5. *Residual Connections*: Skip connections around attention and feed-forward blocks

## Technical Highlights

### Custom Implementations

- *Multi-Head Attention*: Efficient implementation with proper head splitting and causal masking
- *GELU Activation*: Smooth, non-linear activation function using tanh approximation
- *Layer Normalization*: Custom implementation with learnable scale and shift parameters
- *Memory-Mapped Data Loading*: Fast, memory-efficient batch generation from preprocessed binary files

### Training Features

- *AdamW Optimizer*: Weight decay regularization for better generalization
- *Early Stopping*: Automatic termination at end-of-text token during generation
- *Evaluation Metrics*: Periodic train/validation loss tracking
- *Sample Generation*: In-training text generation for monitoring model progress

## Getting Started

### Prerequisites

bash
pip install torch numpy datasets tiktoken tqdm


### Data Preparation

The code automatically downloads and preprocesses the TinyStories dataset:

python
# Downloads dataset from HuggingFace
ds = load_dataset("roneneldan/TinyStories")

# Tokenizes and saves as binary files (train.bin, validation.bin)
# Uses GPT-2 tokenizer with 50,257 vocabulary size


### Training

python
# Initialize model
model = GPT(GPT_CONFIG_124M)
model.to(device)

# Train for 10 epochs
train_losses, val_losses, tokens_seen = train_model_simple(
    model, optimizer, device,
    num_epochs=10, 
    eval_freq=100, 
    eval_iter=5,
    start_context="a boy got a ball"
)


### Generation

python
# Generate text from a prompt
start_context = "a boy was playing then his friend comes"
encoded = text_to_token_ids(start_context, tokenizer).to(device)

generated = model.generate(
    idx=encoded,
    max_new_tokens=300,
    context_size=128,
    temperature=0.7,  # Controls randomness
    top_k=50          # Limits sampling to top-k tokens
)

decoded_text = token_ids_to_text(generated, tokenizer)


## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Batch Size | 32 |
| Context Window | 128 tokens |
| Learning Rate | 0.0004 |
| Weight Decay | 0.1 |
| Epochs | 10 |
| Steps per Epoch | 1,000 |

## Performance

The model learns to generate coherent short stories in the style of the TinyStories dataset. Training convergence is monitored through:

- Train/validation loss curves
- Sample generation at each epoch
- Token-level cross-entropy 



## Key Learning Outcomes

This project demonstrates:

- *Transformer Architecture*: Understanding attention mechanisms, positional encodings, and residual connections
- *Language Modeling*: Autoregressive text generation with teacher forcing
- *PyTorch Best Practices*: Custom modules, efficient data loading, GPU acceleration
- *Training Strategies*: Optimization, regularization, and hyperparameter tuning
- *Production Skills*: Checkpointing, evaluation, and inference pipelines



## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [TinyStories Dataset](https://arxiv.org/abs/2305.07759) - Dataset research paper
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Inspiration for minimal GPT implementation

## License

This project is for educational purposes. The TinyStories dataset is available under its respective license from HuggingFace.

---

*Built with* PyTorch • HuggingFace Datasets • Tiktoken

A hands-on exploration of transformer-based language models
