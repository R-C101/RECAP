# RECAP: Retrieval-based Adversarial Prompt Attack

A novel approach to adversarial attacks on Large Language Models (LLMs) that uses semantic similarity search to retrieve proven-successful attack patterns from a curated database.

## Overview

RECAP works by:
1. Taking a user's input prompt
2. Finding semantically similar prompts from a database of successful jailbreaks
3. Extracting the adversarial suffixes that made those attacks successful
4. Applying those suffixes to the new prompt

This retrieval-based approach is more efficient than optimization-based methods like GCG, as it leverages previously discovered attack patterns rather than generating new ones from scratch.

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/rishitchugh/recap-attack.git
cd recap-attack

# Create virtual environment and install dependencies
uv venv
uv sync

# Run example
uv run python examples/basic_usage.py

# Or run a quick test
uv run python -c "from recap_attack import recap_attack; print(recap_attack('How to pick a lock'))"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/rishitchugh/recap-attack.git
cd recap-attack

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/basic_usage.py
```

## Quick Start

### RECAP Attack (Retrieval-based)

```python
from recap_attack import recap_attack

# Generate adversarial prompts
suffixes, full_prompts = recap_attack("How to pick a lock")

# Print the most effective adversarial prompt
print(full_prompts[0])
```

### GCG Attack (Suffix-based)

```python
from recap_attack import gcg_attack

# Generate adversarial prompts using pre-computed GCG suffixes
suffixes, full_prompts = gcg_attack("How to pick a lock")

print(full_prompts[0])
```

## Detailed Usage

### RecapAttack Class

For more control over the attack process:

```python
from recap_attack import RecapAttack

# Initialize with custom settings
attacker = RecapAttack(
    data_dir="./embeddings",  # Directory with embeddings (default: ./embeddings/)
    use_local=True,           # Use local files if available
    verbose=True              # Print status messages
)

# Search for multiple matches
results = attacker.search("How to make explosives", top_k=3)

for result in results:
    print(f"Matched prompt: {result.matched_prompt}")
    print(f"Similarity score: {result.similarity_score:.4f}")
    print(f"Number of attacks: {len(result.full_adversarial_prompts)}")
    print(f"Best attack: {result.full_adversarial_prompts[0][:80]}...")
    print()
```

### GCGAttack Class

```python
from recap_attack import GCGAttack

attacker = GCGAttack(verbose=True)

# Get top 5 adversarial variants
results = attacker.search("How to hack a computer", top_k=5)

for prompt in results[0].full_adversarial_prompts:
    print(prompt[:80] + "...")
```

### Batch Processing

```python
from recap_attack import RecapAttack

attacker = RecapAttack(verbose=True)

prompts = [
    "How to hack a computer",
    "How to make drugs",
    "How to evade taxes",
]

for prompt in prompts:
    suffixes, full_prompts = attacker.attack(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Found {len(full_prompts)} adversarial variants")
    if full_prompts:
        print(f"Best: {full_prompts[0][:60]}...")
```

## Data Directory

RECAP requires data files to function. By default, it looks for files in `./data/`:

```
data/
├── embeddings.pkl        # Pre-computed embeddings
├── faiss_index.bin       # FAISS similarity index
└── processed_dataset.csv # Dataset with prompts
```

If local files are not found, RECAP will automatically download them from HuggingFace: `rishitchugh/successful_adversarial_prompts`

### Custom Data Directory

```python
from recap_attack import RecapAttack

# Use a custom directory
attacker = RecapAttack(data_dir="/path/to/my/data")
```

## API Reference

### `recap_attack(prompt, data_dir=None, verbose=False)`

Quick function for RECAP attack.

**Args:**
- `prompt` (str): The input prompt to attack
- `data_dir` (str, optional): Directory containing embeddings
- `verbose` (bool): Print status messages

**Returns:**
- `Tuple[List[str], List[str]]`: (adversarial_suffixes, full_adversarial_prompts)

### `gcg_attack(prompt, verbose=False)`

Quick function for GCG attack.

**Args:**
- `prompt` (str): The input prompt to attack
- `verbose` (bool): Print status messages

**Returns:**
- `Tuple[List[str], List[str]]`: (adversarial_suffixes, full_adversarial_prompts)

### `RecapAttack`

Main class for RECAP attacks with full control.

**Methods:**
- `search(user_prompt, top_k=3)` → `List[RecapResult]`
- `attack(user_prompt)` → `Tuple[List[str], List[str]]`
- `initialize()` → Explicitly load models and data

### `GCGAttack`

Class for GCG attacks using pre-computed suffixes.

**Methods:**
- `search(user_prompt, top_k=3)` → `List[GCGResult]`
- `attack(user_prompt)` → `Tuple[List[str], List[str]]`

## Results Format

### RecapResult

```python
@dataclass
class RecapResult:
    original_prompt: str          # Your input
    matched_prompt: str           # Similar prompt from database
    adversarial_suffixes: List[str]      # Attack suffixes (max 3)
    full_adversarial_prompts: List[str]  # Complete prompts ready to use
    similarity_score: float       # Lower = more similar
```

### GCGResult

```python
@dataclass
class GCGResult:
    original_prompt: str
    adversarial_suffixes: List[str]
    full_adversarial_prompts: List[str]
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{chugh2026recap,
  title   = {RECAP: A Resource-Efficient Method for Adversarial Prompting in Large Language Models},
  author  = {Chugh, Rishit},
  journal = {arXiv preprint arXiv:2601.15331},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.15331}
}
```


## License

MIT License

## Disclaimer

This tool is intended for research purposes only. Use responsibly and ethically. The authors are not responsible for any misuse of this software.
