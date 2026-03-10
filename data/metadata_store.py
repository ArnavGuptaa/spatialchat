"""
Dataset metadata store — pre-computed gene lists and cell types per dataset.

Stored as JSON files in data/metadata/{dataset_id}.json so that tools can
quickly look up gene names and cell types without loading the full AnnData
into memory. Updated automatically by the ingestion script.

Schema per dataset:
{
    "dataset_id": "mouse_brain_seqfish",
    "genes": ["Abcc4", "Acp5", ...],          # full sorted gene list
    "celltypes": ["Astrocytes", "Neurons", ...], # unique cell types (if available)
    "n_cells": 19416,
    "n_genes": 351,
    "annotations": ["celltype_mapped_refined"],
    "celltype_column": "celltype_mapped_refined"
}
"""

from __future__ import annotations

import json
import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

METADATA_DIR = Path(__file__).parent / "metadata"


def _ensure_metadata_dir():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def get_metadata_path(dataset_id: str) -> Path:
    """Return the path to a dataset's metadata JSON file."""
    return METADATA_DIR / f"{dataset_id}.json"


def load_metadata(dataset_id: str) -> Optional[dict]:
    """Load metadata for a dataset, or None if not available."""
    path = get_metadata_path(dataset_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_metadata(dataset_id: str, metadata: dict) -> Path:
    """Save metadata for a dataset. Returns the file path."""
    _ensure_metadata_dir()
    path = get_metadata_path(dataset_id)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata for '{dataset_id}' to {path}")
    return path


def build_metadata_from_adata(dataset_id: str, adata, celltype_col: Optional[str] = None) -> dict:
    """
    Extract metadata from an AnnData object.

    Args:
        dataset_id: The dataset identifier
        adata: An AnnData object
        celltype_col: Optional column name in obs containing cell types

    Returns:
        Metadata dict ready to save with save_metadata()
    """
    genes = sorted(adata.var_names.tolist())

    celltypes = []
    if celltype_col and celltype_col in adata.obs.columns:
        cats = adata.obs[celltype_col]
        if hasattr(cats, "cat"):
            celltypes = sorted(cats.cat.categories.tolist())
        else:
            celltypes = sorted(cats.unique().tolist())

    return {
        "dataset_id": dataset_id,
        "genes": genes,
        "celltypes": celltypes,
        "n_cells": adata.shape[0],
        "n_genes": adata.shape[1],
        "annotations": list(adata.obs.columns),
        "celltype_column": celltype_col or "",
    }


# ── Query helpers (used by tools) ────────────────────────────


def get_gene_list(dataset_id: str) -> Optional[list[str]]:
    """Get the full gene list for a dataset from metadata."""
    meta = load_metadata(dataset_id)
    if meta is None:
        return None
    return meta.get("genes", [])


def get_celltypes(dataset_id: str) -> Optional[list[str]]:
    """Get the cell type list for a dataset from metadata."""
    meta = load_metadata(dataset_id)
    if meta is None:
        return None
    return meta.get("celltypes", [])


def find_similar_genes(dataset_id: str, query: str, n: int = 5) -> list[str]:
    """
    Find genes similar to the query using fuzzy matching.

    Uses difflib.get_close_matches for fast approximate string matching
    on the pre-computed gene list. Falls back to substring matching.
    """
    genes = get_gene_list(dataset_id)
    if not genes:
        return []

    # 1. Exact case-insensitive match
    lower = query.lower()
    exact = [g for g in genes if g.lower() == lower]
    if exact:
        return exact

    # 2. Fuzzy match (edit distance)
    # get_close_matches is case-sensitive, so we build a lowercase map
    lower_to_orig = {}
    lower_genes = []
    for g in genes:
        gl = g.lower()
        lower_to_orig[gl] = g
        lower_genes.append(gl)

    fuzzy = get_close_matches(lower, lower_genes, n=n, cutoff=0.6)
    if fuzzy:
        return [lower_to_orig[f] for f in fuzzy]

    # 3. Substring match
    substr = [g for g in genes if lower in g.lower()][:n]
    if substr:
        return substr

    # 4. Prefix match
    prefix = [g for g in genes if g.lower().startswith(lower[:3])][:n]
    return prefix
