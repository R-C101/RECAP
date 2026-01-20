"""
RECAP: Retrieval-based Adversarial Prompt Attack

A novel approach to adversarial attacks on LLMs using semantic similarity search
to retrieve proven-successful attack patterns from a curated database.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path
from huggingface_hub import hf_hub_download
from dataclasses import dataclass


@dataclass
class RecapResult:
    """Container for RECAP attack results."""
    original_prompt: str
    matched_prompt: str
    adversarial_suffixes: List[str]
    full_adversarial_prompts: List[str]
    similarity_score: float


class RecapAttack:
    """
    RECAP (Retrieval-based Adversarial Prompt) Attack
    
    Uses semantic similarity search to find successful adversarial prompts
    from a curated dataset and adapts them to new inputs.
    
    Attributes:
        model: SentenceTransformer model for embeddings
        index: FAISS index for similarity search
        embeddings: Precomputed embeddings
        adv_values: Adversarial prompt values
        labels: Success labels for each adversarial prompt
        prompts: Original prompts from dataset
    """
    
    REPO_ID = "rishitchugh/successful_adversarial_prompts"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_LOCAL_DIR = "data"
    
    def __init__(
        self, 
        data_dir: Optional[str] = None, 
        use_local: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the RECAP attack.
        
        Args:
            data_dir: Directory containing embeddings.pkl, faiss_index.bin, 
                      and processed_dataset.csv. If None, uses './embeddings/'
            use_local: If True, look for local files first. If False or local 
                       files not found, download from HuggingFace
            verbose: Whether to print status messages
        """
        self.verbose = verbose
        self.use_local = use_local
        
        # Set data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(self.DEFAULT_LOCAL_DIR)
        
        self._model = None
        self._index = None
        self._embeddings = None
        self._adv_values = None
        self._labels = None
        self._prompts = None
        self._initialized = False
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[RECAP] {message}")
    
    def _get_local_paths(self) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Get paths to local files if they exist."""
        csv_path = self.data_dir / "processed_dataset.csv"
        index_path = self.data_dir / "faiss_index.bin"
        embeddings_path = self.data_dir / "embeddings.pkl"
        
        if csv_path.exists() and index_path.exists() and embeddings_path.exists():
            return csv_path, index_path, embeddings_path
        return None, None, None
    
    def _download_files(self) -> Tuple[str, str, str]:
        """Download required files from HuggingFace Hub."""
        self._log("Downloading dataset files from HuggingFace...")
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        kwargs = {"repo_id": self.REPO_ID, "repo_type": "dataset"}
        
        csv_path = hf_hub_download(
            filename="processed_dataset.csv", 
            local_dir=str(self.data_dir),
            **kwargs
        )
        index_path = hf_hub_download(
            filename="faiss_index.bin", 
            local_dir=str(self.data_dir),
            **kwargs
        )
        embeddings_path = hf_hub_download(
            filename="embeddings.pkl", 
            local_dir=str(self.data_dir),
            **kwargs
        )
        
        return csv_path, index_path, embeddings_path
    
    def _load_files(self) -> Tuple[str, str, str]:
        """Load files from local directory or download from HuggingFace."""
        if self.use_local:
            local_paths = self._get_local_paths()
            if all(local_paths):
                self._log(f"Using local files from: {self.data_dir}")
                return tuple(str(p) for p in local_paths)
            else:
                self._log(f"Local files not found in {self.data_dir}, downloading...")
        
        return self._download_files()
    
    def initialize(self):
        """
        Initialize the attack by loading models and data.
        Call this explicitly or it will be called automatically on first search.
        """
        if self._initialized:
            return
        
        csv_path, index_path, embeddings_path = self._load_files()
        
        # Load embedding model
        self._log(f"Loading embedding model: {self.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(self.EMBEDDING_MODEL)
        
        # Load FAISS index
        self._log("Loading FAISS index...")
        self._index = faiss.read_index(index_path)
        
        # Load embeddings and labels
        self._log("Loading embeddings and labels...")
        with open(embeddings_path, 'rb') as f:
            self._embeddings, self._adv_values, self._labels = pickle.load(f)
        
        # Load prompts
        self._log("Loading prompt dataset...")
        df = pd.read_csv(csv_path)
        self._prompts = df['Pez Prompt'].tolist()
        
        self._initialized = True
        self._log("Initialization complete!")
    
    def _clean_suffix(self, suffix: str, original_prompt: str) -> str:
        """Clean adversarial suffix by removing original prompt and artifacts."""
        cleaned = suffix.replace(original_prompt, '')
        cleaned = cleaned.replace('[/INST]', '')
        cleaned = cleaned.replace('\n\n', ' ')
        return cleaned.strip()
    
    def search(
        self, 
        user_prompt: str, 
        top_k: int = 3
    ) -> List[RecapResult]:
        """
        Search for adversarial prompts similar to the user's input.
        
        Args:
            user_prompt: The input prompt to find adversarial versions for
            top_k: Number of top matches to return (default: 3)
            
        Returns:
            List of RecapResult objects containing:
                - original_prompt: The user's input
                - matched_prompt: The similar prompt found in database
                - adversarial_suffixes: The attack suffixes (ordered by success rate)
                - full_adversarial_prompts: Complete adversarial prompts ready to use
                - similarity_score: How similar the match is (lower = more similar)
        """
        if not self._initialized:
            self.initialize()
        
        # Compute embedding for user input
        user_embedding = self._model.encode([user_prompt], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self._index.search(user_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            matched_prompt = self._prompts[idx]
            advs = self._adv_values[idx]
            lbls = self._labels[idx]
            
            # Filter to only successful adversarial prompts (label == 1)
            successful_advs = []
            for adv, lbl in zip(advs, lbls):
                if lbl == 1:
                    successful_advs.append(adv)
            
            if not successful_advs:
                continue
            
            # Clean suffixes and create full prompts
            suffixes = []
            full_prompts = []
            for adv in successful_advs[:3]:  # Top 3 by success rate
                suffix = self._clean_suffix(adv, matched_prompt)
                suffixes.append(suffix)
                full_prompts.append(f"{user_prompt} {suffix}")
            
            results.append(RecapResult(
                original_prompt=user_prompt,
                matched_prompt=matched_prompt,
                adversarial_suffixes=suffixes,
                full_adversarial_prompts=full_prompts,
                similarity_score=float(distances[0][i])
            ))
        
        return results
    
    def attack(self, user_prompt: str) -> Tuple[List[str], List[str]]:
        """
        Simple interface to get adversarial prompts.
        
        Args:
            user_prompt: The input prompt to attack
            
        Returns:
            Tuple of (adversarial_suffixes, full_adversarial_prompts)
            Both lists are ordered by success rate (best first)
        """
        results = self.search(user_prompt, top_k=1)
        
        if not results:
            return [], []
        
        best_match = results[0]
        return best_match.adversarial_suffixes, best_match.full_adversarial_prompts


# Convenience function for quick usage
def recap_attack(
    prompt: str, 
    data_dir: Optional[str] = None,
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Quick function to generate adversarial prompts using RECAP.
    
    Args:
        prompt: The input prompt to attack
        data_dir: Optional directory containing embeddings (default: ./embeddings/)
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (adversarial_suffixes, full_adversarial_prompts)
        
    Example:
        >>> from recap_attack import recap_attack
        >>> suffixes, full_prompts = recap_attack("How to pick a lock")
        >>> print(full_prompts[0])  # Most likely to succeed
    """
    attacker = RecapAttack(data_dir=data_dir, verbose=verbose)
    return attacker.attack(prompt)
