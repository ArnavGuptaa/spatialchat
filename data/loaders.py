"""
Universal dataset loading from h5ad files.

Handles dataset discovery from a catalog, caching in memory with LRU
eviction, and loading from h5ad files with validation.

Key design:
  - No per-dataset loader functions — everything loads via h5ad
  - Catalog specifies data_file path and celltype column
  - DatasetCache manages LRU in-memory caching
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import anndata as ad

from config.settings import get_settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────
# Catalog
# ──────────────────────────────────────────────────────────────

def load_catalog() -> dict:
    """Load the dataset catalog from catalog.json."""
    catalog_path = DATA_DIR / "catalog.json"
    with open(catalog_path) as f:
        return json.load(f)


def list_datasets() -> list[dict]:
    """Return a list of dataset summaries for the Dataset Finder agent."""
    catalog = load_catalog()
    summaries = []
    for dataset_id, info in catalog.items():
        summaries.append({
            "id": dataset_id,
            "display_name": info["display_name"],
            "tissue": info["tissue"],
            "species": info["species"],
            "technology": info["technology"],
            "n_spots_approx": info.get("n_spots_approx", "unknown"),
        })
    return summaries


def get_celltype_column(dataset_id: str) -> Optional[str]:
    """Get the celltype obs column name for a dataset from the catalog."""
    catalog = load_catalog()
    info = catalog.get(dataset_id, {})
    return info.get("celltype")


# ──────────────────────────────────────────────────────────────
# Dataset Cache Manager
# ──────────────────────────────────────────────────────────────

class DatasetCache:
    """
    Manages loaded AnnData objects in memory with LRU-style eviction.

    Spatial datasets are large (1-5 GB in memory). This class holds up
    to max_size datasets and evicts the least recently used when a new
    one is loaded. Uses OrderedDict for O(1) LRU tracking.
    """

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._store: OrderedDict[str, ad.AnnData] = OrderedDict()

    def get(self, dataset_id: str) -> Optional[ad.AnnData]:
        """Retrieve a loaded dataset, or None if not cached."""
        if dataset_id in self._store:
            # Move to end = most recently used
            self._store.move_to_end(dataset_id)
            return self._store[dataset_id]
        return None

    def put(self, dataset_id: str, adata: ad.AnnData) -> None:
        """Store a dataset, evicting LRU if at capacity."""
        if dataset_id in self._store:
            self._store.move_to_end(dataset_id)
            self._store[dataset_id] = adata
        else:
            if len(self._store) >= self.max_size:
                evicted_id, _ = self._store.popitem(last=False)
                logger.info(f"Evicted dataset '{evicted_id}' from cache (LRU)")
            self._store[dataset_id] = adata
            logger.info(f"Cached dataset '{dataset_id}' ({len(self._store)}/{self.max_size})")

    def is_loaded(self, dataset_id: str) -> bool:
        return dataset_id in self._store

    def loaded_ids(self) -> list[str]:
        return list(self._store.keys())


# Singleton cache instance
_cache = DatasetCache(max_size=get_settings().max_loaded_datasets)


def get_cache() -> DatasetCache:
    return _cache


# ──────────────────────────────────────────────────────────────
# Universal Loader
# ──────────────────────────────────────────────────────────────

def load_dataset(dataset_id: str) -> ad.AnnData:
    """
    Load a dataset by ID from its h5ad file.

    Uses cache-aside pattern: check cache first, load h5ad on miss,
    store result for future access. Validates spatial coordinates exist.
    """
    # Check cache first
    cached = _cache.get(dataset_id)
    if cached is not None:
        logger.info(f"Dataset '{dataset_id}' served from cache")
        return cached

    # Validate dataset_id exists in catalog
    catalog = load_catalog()
    if dataset_id not in catalog:
        available = list(catalog.keys())
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Available datasets: {available}"
        )

    info = catalog[dataset_id]

    # Resolve data file path
    data_file = info.get("data_file")
    if not data_file:
        raise ValueError(
            f"Dataset '{dataset_id}' has no 'data_file' in catalog. "
            f"Use the ingestion script to add it."
        )

    data_path = DATA_DIR / data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Use the ingestion script to add the h5ad file."
        )

    # Load h5ad
    logger.info(f"Loading '{dataset_id}' from {data_path}...")
    adata = ad.read_h5ad(data_path)

    # Validate spatial coordinates
    if "spatial" not in adata.obsm:
        raise ValueError(
            f"Dataset '{dataset_id}' is missing 'spatial' in obsm. "
            f"Spatial coordinates are required for SpatialChat."
        )

    # Validate celltype column if specified
    celltype_col = info.get("celltype")
    if celltype_col and celltype_col not in adata.obs.columns:
        logger.warning(
            f"Dataset '{dataset_id}': celltype column '{celltype_col}' "
            f"not found in obs. Available: {list(adata.obs.columns)[:10]}"
        )

    logger.info(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    _cache.put(dataset_id, adata)
    return adata
