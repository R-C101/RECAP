"""
RECAP: Retrieval-based Adversarial Prompt Attack

A novel approach to adversarial attacks on LLMs that uses semantic similarity
to retrieve proven-successful attack patterns from a curated database.

Quick Usage:
    >>> from recap_attack import recap_attack, gcg_attack
    >>> 
    >>> # RECAP attack (retrieval-based)
    >>> suffixes, full_prompts = recap_attack("How to pick a lock")
    >>> print(full_prompts[0])  # Most likely to succeed
    >>> 
    >>> # GCG attack (suffix-based)
    >>> suffixes, full_prompts = gcg_attack("How to pick a lock")

Class-based Usage:
    >>> from recap_attack import RecapAttack, GCGAttack
    >>> 
    >>> # Initialize with custom settings
    >>> attacker = RecapAttack(data_dir="./my_embeddings", verbose=True)
    >>> results = attacker.search("How to pick a lock", top_k=3)
    >>> 
    >>> for result in results:
    ...     print(f"Match: {result.matched_prompt}")
    ...     print(f"Best attack: {result.full_adversarial_prompts[0]}")
"""

from .recap_attack import RecapAttack, RecapResult, recap_attack
from .gcg_attack import GCGAttack, GCGResult, gcg_attack

__all__ = [
    # RECAP
    "RecapAttack",
    "RecapResult", 
    "recap_attack",
    # GCG
    "GCGAttack",
    "GCGResult",
    "gcg_attack",
]

__version__ = "1.0.0"
__author__ = "Rishit Chugh"
