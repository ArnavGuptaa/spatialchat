"""
Dataset metadata store — gene lists, cell types, and vector search per dataset.

Primary store is now ChromaDB (via ``data.vector_store``), which holds gene
expression embeddings and cell type metadata.  JSON files in
``data/metadata/{dataset_id}.json`` are kept as a lightweight fallback for
environments where ChromaDB is not installed or not yet indexed.

Schema per JSON file:
{
    "dataset_id": "mouse_brain_seqfish",
    "genes": ["Abcc4", "Acp5", ...],
    "celltypes": ["Astrocytes", "Neurons", ...],
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


# ---------------------------------------------------------------------------
# ChromaDB helpers (lazy import so the module works without chromadb)
# ---------------------------------------------------------------------------

def _get_vs():
    """Return the VectorStore singleton, or None if unavailable."""
    try:
        from data.vector_store import get_vector_store
        return get_vector_store()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON persistence (kept as fallback)
# ---------------------------------------------------------------------------

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

    Also triggers ChromaDB indexing (gene expression embeddings + celltype
    metadata) when the vector store is available.

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

    metadata = {
        "dataset_id": dataset_id,
        "genes": genes,
        "celltypes": celltypes,
        "n_cells": adata.shape[0],
        "n_genes": adata.shape[1],
        "annotations": list(adata.obs.columns),
        "celltype_column": celltype_col or "",
    }

    # Index in ChromaDB (gene expression embeddings + celltype metadata)
    vs = _get_vs()
    if vs is not None:
        try:
            result = vs.index_dataset(dataset_id, adata, celltype_col)
            logger.info(
                f"ChromaDB indexed {result['n_genes_indexed']} genes, "
                f"{result['n_celltypes_indexed']} celltypes for '{dataset_id}'"
            )
        except Exception as e:
            logger.warning(f"ChromaDB indexing failed for '{dataset_id}': {e}")

    return metadata


# ---------------------------------------------------------------------------
# Query helpers (used by tools) — ChromaDB primary, JSON fallback
# ---------------------------------------------------------------------------


def get_gene_list(dataset_id: str) -> Optional[list[str]]:
    """Get the full gene list for a dataset (ChromaDB first, then JSON)."""
    vs = _get_vs()
    if vs is not None and vs.is_indexed(dataset_id):
        genes = vs.get_gene_list(dataset_id)
        if genes:
            return genes

    # Fallback to JSON
    meta = load_metadata(dataset_id)
    if meta is None:
        return None
    return meta.get("genes", [])


def get_celltypes(dataset_id: str) -> Optional[list[str]]:
    """Get the cell type list for a dataset (ChromaDB first, then JSON)."""
    vs = _get_vs()
    if vs is not None and vs.is_indexed(dataset_id):
        cts = vs.get_celltypes(dataset_id)
        if cts:
            return cts

    # Fallback to JSON
    meta = load_metadata(dataset_id)
    if meta is None:
        return None
    return meta.get("celltypes", [])


def find_similar_genes(dataset_id: str, query: str, n: int = 5) -> list[str]:
    """
    Find genes similar to the query using fuzzy matching.

    Tries ChromaDB text search first for semantic matching, then falls back
    to difflib-based fuzzy matching on the gene list.
    """
    # --- ChromaDB text search (semantic) ---
    vs = _get_vs()
    if vs is not None and vs.is_indexed(dataset_id):
        try:
            results = vs.search_genes_by_text(dataset_id, query, n=n)
            if results:
                return [r["gene_symbol"] for r in results]
        except Exception as e:
            logger.debug(f"ChromaDB text search failed, falling back to difflib: {e}")

    # --- Fallback: difflib fuzzy matching ---
    genes = get_gene_list(dataset_id)
    if not genes:
        return []

    # 1. Exact case-insensitive match
    lower = query.lower()
    exact = [g for g in genes if g.lower() == lower]
    if exact:
        return exact

    # 2. Fuzzy match (edit distance)
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


# ---------------------------------------------------------------------------
# New semantic / RAG search functions
# ---------------------------------------------------------------------------


def search_genes_semantic(dataset_id: str, query: str, n: int = 10) -> list[dict]:
    """
    Search genes semantically using ChromaDB document search.

    Returns a list of dicts with gene_symbol, mean_expression,
    pct_expressing, distance.

    Example:
        search_genes_semantic("mouse_brain_seqfish", "highly expressed")
    """
    vs = _get_vs()
    if vs is None:
        return []
    return vs.search_genes_by_text(dataset_id, query, n)


def find_expression_similar_genes(dataset_id: str, gene: str, n: int = 10) -> list[dict]:
    """
    Find genes with similar expression profiles via vector similarity search.

    Uses the gene expression embedding stored in ChromaDB to find genes
    whose statistical profiles (mean, median, pct_expressing, per-celltype
    means) are closest in cosine distance.

    Args:
        dataset_id: Dataset to search.
        gene: Reference gene name.
        n: Number of similar genes to return.

    Returns:
        List of dicts with gene_symbol, distance, mean_expression,
        pct_expressing, top_celltypes.
    """
    vs = _get_vs()
    if vs is None:
        return []
    return vs.search_similar_genes(dataset_id, gene, n)


def search_celltypes_semantic(dataset_id: str, query: str, n: int = 5) -> list[dict]:
    """
    Search cell types by keyword using ChromaDB.

    Returns list of dicts with celltype, n_cells, marker_genes.
    """
    vs = _get_vs()
    if vs is None:
        return []
    return vs.search_celltypes(dataset_id, query, n)
