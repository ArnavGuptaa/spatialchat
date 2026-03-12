"""
ChromaDB vector store for gene expression RAG.

Replaces JSON-based metadata caching with a persistent vector database
that enables semantic search over gene expression profiles and cell type
metadata. Gene expression statistics are embedded as fixed-size vectors,
allowing similarity search (e.g. "find genes with similar expression
patterns to Snap25").

Collections per dataset:
  - gene_metadata_{dataset_id}: gene-level stats + expression embeddings
  - celltype_metadata_{dataset_id}: cell type info with marker genes

Embedding strategy:
  Each gene gets a numeric vector built from summary statistics:
  [log1p(mean), log1p(median), pct_expressing/100, log1p(std),
   per_celltype_mean_expression_0, ..., per_celltype_mean_expression_N]
  Normalised to unit length so cosine similarity captures expression
  profile shape, not magnitude.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

VECTORDB_DIR = Path(__file__).parent / "vectordb"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _build_gene_embedding(
    mean_expr: float,
    median_expr: float,
    pct_expressing: float,
    std_expr: float,
    celltype_means: list[float],
) -> list[float]:
    """
    Build a fixed-size embedding vector from gene expression statistics.

    The vector captures overall expression level plus per-celltype profile,
    enabling similarity search that groups co-regulated genes together.

    Args:
        mean_expr: Mean expression across all cells.
        median_expr: Median expression across all cells.
        pct_expressing: Percentage of cells with expression > 0.
        std_expr: Standard deviation of expression.
        celltype_means: Mean expression per cell type (ordered consistently).

    Returns:
        L2-normalised float list suitable as a ChromaDB embedding.
    """
    vec = [
        float(np.log1p(mean_expr)),
        float(np.log1p(median_expr)),
        float(pct_expressing / 100.0),
        float(np.log1p(std_expr)),
    ] + [float(np.log1p(v)) for v in celltype_means]

    # L2 normalise so cosine distance captures profile shape
    arr = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def _compute_gene_stats(adata, gene_idx: int, celltype_col: str | None, celltypes: list[str]) -> dict:
    """
    Compute expression statistics for a single gene.

    Args:
        adata: AnnData object.
        gene_idx: Column index of the gene in adata.X.
        celltype_col: obs column with cell type labels (or None).
        celltypes: Ordered list of unique cell types for consistent embedding dims.

    Returns:
        Dict with gene_symbol, mean, median, std, pct_expressing,
        celltype_means (ordered list), and top_celltypes (JSON str).
    """
    expr_col = adata.X[:, gene_idx]
    if sparse.issparse(expr_col):
        expr = np.asarray(expr_col.todense()).flatten()
    else:
        expr = np.asarray(expr_col).flatten()

    mean_val = float(np.mean(expr))
    median_val = float(np.median(expr))
    std_val = float(np.std(expr))
    pct_expressing = float((expr > 0).mean() * 100)

    # Per-celltype means
    celltype_means: list[float] = []
    top_celltypes: dict[str, float] = {}

    if celltype_col and celltype_col in adata.obs.columns:
        labels = adata.obs[celltype_col].values
        for ct in celltypes:
            mask = labels == ct
            if mask.any():
                ct_mean = float(np.mean(expr[mask]))
            else:
                ct_mean = 0.0
            celltype_means.append(ct_mean)
            top_celltypes[ct] = round(ct_mean, 4)

        # Keep only top 5 for metadata
        top_celltypes = dict(
            sorted(top_celltypes.items(), key=lambda x: x[1], reverse=True)[:5]
        )

    return {
        "gene_symbol": str(adata.var_names[gene_idx]),
        "mean_expr": mean_val,
        "median_expr": median_val,
        "std_expr": std_val,
        "pct_expressing": pct_expressing,
        "n_cells": int(adata.shape[0]),
        "celltype_means": celltype_means,
        "top_celltypes_json": json.dumps(top_celltypes),
    }


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB-backed vector store for gene expression metadata and embeddings.

    Runs in embedded/persistent mode — no separate server needed. Data is
    stored on disk under ``data/vectordb/`` and survives restarts.
    """

    def __init__(self, persist_dir: Path | None = None):
        import chromadb

        self._persist_dir = persist_dir or VECTORDB_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        # Try persistent client first; fall back to ephemeral (in-memory)
        # if the filesystem doesn't support SQLite locking (e.g. some
        # mounted/network filesystems).
        try:
            self.client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._persistent = True
            logger.info(f"ChromaDB initialised (persistent) at {self._persist_dir}")
        except Exception as e:
            logger.warning(
                f"Persistent ChromaDB failed ({e}); falling back to EphemeralClient. "
                f"Data will not survive restarts."
            )
            self.client = chromadb.EphemeralClient()
            self._persistent = False

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _gene_collection_name(self, dataset_id: str) -> str:
        return f"gene_metadata_{dataset_id}"

    def _celltype_collection_name(self, dataset_id: str) -> str:
        return f"celltype_metadata_{dataset_id}"

    def _get_or_create(self, name: str, metadata: dict | None = None):
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_dataset(
        self,
        dataset_id: str,
        adata,
        celltype_col: str | None = None,
    ) -> dict:
        """
        Compute gene expression statistics and index everything in ChromaDB.

        This is the main entry point called during dataset ingestion. It:
          1. Computes per-gene summary statistics from the expression matrix.
          2. Builds a numeric embedding per gene from those statistics.
          3. Upserts genes into the ``gene_metadata_{dataset_id}`` collection.
          4. Computes per-celltype marker genes and upserts into
             ``celltype_metadata_{dataset_id}``.

        Args:
            dataset_id: Unique dataset identifier (e.g. "mouse_brain_seqfish").
            adata: Loaded AnnData object with expression matrix and obs.
            celltype_col: Column in ``adata.obs`` containing cell type labels.

        Returns:
            Summary dict with ``n_genes_indexed`` and ``n_celltypes_indexed``.
        """
        # Resolve ordered celltype list for consistent embedding dims
        celltypes: list[str] = []
        if celltype_col and celltype_col in adata.obs.columns:
            cats = adata.obs[celltype_col]
            if hasattr(cats, "cat"):
                celltypes = sorted(cats.cat.categories.tolist())
            else:
                celltypes = sorted(cats.unique().tolist())

        # ---- Gene collection ----
        gene_col = self._get_or_create(self._gene_collection_name(dataset_id))

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []

        n_genes = adata.shape[1]
        for idx in range(n_genes):
            stats = _compute_gene_stats(adata, idx, celltype_col, celltypes)
            gene_sym = stats["gene_symbol"]

            ids.append(f"{dataset_id}__{gene_sym}")
            documents.append(
                f"{gene_sym}: mean={stats['mean_expr']:.4f}, "
                f"median={stats['median_expr']:.4f}, "
                f"{stats['pct_expressing']:.1f}% expressing. "
                f"Top celltypes: {stats['top_celltypes_json']}"
            )
            metadatas.append({
                "gene_symbol": gene_sym,
                "dataset_id": dataset_id,
                "mean_expression": stats["mean_expr"],
                "median_expression": stats["median_expr"],
                "std_expression": stats["std_expr"],
                "pct_expressing": stats["pct_expressing"],
                "n_cells": stats["n_cells"],
                "top_celltypes": stats["top_celltypes_json"],
            })
            embeddings.append(
                _build_gene_embedding(
                    stats["mean_expr"],
                    stats["median_expr"],
                    stats["pct_expressing"],
                    stats["std_expr"],
                    stats["celltype_means"],
                )
            )

        # Upsert in batches (ChromaDB recommends <=5 000 per call)
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            gene_col.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        logger.info(f"Indexed {n_genes} genes for '{dataset_id}'")

        # ---- Celltype collection ----
        n_celltypes = 0
        if celltypes:
            ct_col = self._get_or_create(
                self._celltype_collection_name(dataset_id),
                metadata={"hnsw:space": "cosine"},
            )

            ct_ids: list[str] = []
            ct_docs: list[str] = []
            ct_metas: list[dict] = []

            labels = adata.obs[celltype_col].values
            for ct in celltypes:
                mask = labels == ct
                n_cells_ct = int(mask.sum())

                # Marker genes: top 10 genes by mean expression in this celltype
                expr_sub = adata[mask, :].X
                mean_by_gene = np.asarray(expr_sub.mean(axis=0)).flatten()
                top_idx = np.argsort(mean_by_gene)[::-1][:10]
                marker_genes = [str(adata.var_names[g]) for g in top_idx]

                ct_ids.append(f"{dataset_id}__{ct}")
                ct_docs.append(
                    f"{ct}: {n_cells_ct} cells. "
                    f"Marker genes: {', '.join(marker_genes[:5])}"
                )
                ct_metas.append({
                    "celltype": ct,
                    "dataset_id": dataset_id,
                    "n_cells": n_cells_ct,
                    "marker_genes": json.dumps(marker_genes),
                    "annotation_column": celltype_col or "",
                })

            ct_col.upsert(ids=ct_ids, documents=ct_docs, metadatas=ct_metas)
            n_celltypes = len(celltypes)
            logger.info(f"Indexed {n_celltypes} cell types for '{dataset_id}'")

        return {"n_genes_indexed": n_genes, "n_celltypes_indexed": n_celltypes}

    # ------------------------------------------------------------------
    # Retrieval — gene similarity (vector search)
    # ------------------------------------------------------------------

    def search_similar_genes(
        self,
        dataset_id: str,
        query_gene: str,
        n: int = 10,
    ) -> list[dict]:
        """
        Find genes with similar expression profiles using vector similarity.

        Looks up the embedding of ``query_gene`` and runs a nearest-neighbour
        search in ChromaDB to find genes with a similar statistical profile
        across cell types.

        Args:
            dataset_id: Dataset to search in.
            query_gene: Reference gene name (must already be indexed).
            n: Number of results.

        Returns:
            List of dicts with gene_symbol, distance, and expression stats.
        """
        col_name = self._gene_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
        except Exception:
            return []

        gene_id = f"{dataset_id}__{query_gene}"
        try:
            existing = collection.get(ids=[gene_id], include=["embeddings"])
        except Exception:
            return []

        if not existing["ids"]:
            return []

        query_emb = existing["embeddings"][0]
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n + 1,  # +1 to exclude the query gene itself
        )

        output: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i, gid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                if meta["gene_symbol"] == query_gene:
                    continue  # skip self
                output.append({
                    "gene_symbol": meta["gene_symbol"],
                    "distance": float(results["distances"][0][i]),
                    "mean_expression": meta["mean_expression"],
                    "pct_expressing": meta["pct_expressing"],
                    "top_celltypes": meta.get("top_celltypes", "{}"),
                })
                if len(output) >= n:
                    break

        return output

    # ------------------------------------------------------------------
    # Retrieval — text / keyword search
    # ------------------------------------------------------------------

    def search_genes_by_text(
        self,
        dataset_id: str,
        query_text: str,
        n: int = 10,
    ) -> list[dict]:
        """
        Search genes by keyword against their document text and metadata.

        Because the gene collection uses custom numeric embeddings (not text
        embeddings), we cannot use ``query_texts``. Instead we retrieve all
        documents and perform a lightweight keyword match in Python. This is
        fast enough for datasets with <50 000 genes.

        Args:
            dataset_id: Dataset to search in.
            query_text: Free-text query (e.g. "endothelium", "highly expressed").
            n: Number of results.

        Returns:
            List of dicts with gene_symbol and expression stats.
        """
        col_name = self._gene_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
        except Exception:
            return []

        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return []

        # Score each gene by keyword overlap
        query_tokens = set(query_text.lower().split())
        scored: list[tuple[float, int]] = []
        for idx, doc in enumerate(all_data["documents"]):
            doc_lower = doc.lower()
            score = sum(1 for t in query_tokens if t in doc_lower)
            # Also check metadata fields (top_celltypes, gene_symbol)
            meta = all_data["metadatas"][idx]
            meta_text = f"{meta.get('gene_symbol', '')} {meta.get('top_celltypes', '')}".lower()
            score += sum(1 for t in query_tokens if t in meta_text)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        output: list[dict] = []
        for score, idx in scored[:n]:
            meta = all_data["metadatas"][idx]
            output.append({
                "gene_symbol": meta["gene_symbol"],
                "distance": None,
                "mean_expression": meta["mean_expression"],
                "pct_expressing": meta["pct_expressing"],
                "top_celltypes": meta.get("top_celltypes", "{}"),
            })
        return output

    def search_celltypes(
        self,
        dataset_id: str,
        query_text: str,
        n: int = 5,
    ) -> list[dict]:
        """
        Search cell types by keyword.

        Uses keyword matching against celltype documents and metadata.

        Args:
            dataset_id: Dataset to search.
            query_text: Free-text query (e.g. "neurons", "endothelium").
            n: Number of results.

        Returns:
            List of dicts with celltype, n_cells, marker_genes.
        """
        col_name = self._celltype_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
        except Exception:
            return []

        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return []

        query_tokens = set(query_text.lower().split())
        scored: list[tuple[float, int]] = []
        for idx, doc in enumerate(all_data["documents"]):
            doc_lower = doc.lower()
            score = sum(1 for t in query_tokens if t in doc_lower)
            meta = all_data["metadatas"][idx]
            meta_text = f"{meta.get('celltype', '')} {meta.get('marker_genes', '')}".lower()
            score += sum(1 for t in query_tokens if t in meta_text)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        output: list[dict] = []
        for score, idx in scored[:n]:
            meta = all_data["metadatas"][idx]
            output.append({
                "celltype": meta["celltype"],
                "n_cells": meta["n_cells"],
                "marker_genes": meta.get("marker_genes", "[]"),
                "annotation_column": meta.get("annotation_column", ""),
            })
        return output

    # ------------------------------------------------------------------
    # Metadata helpers (replace JSON gene list lookups)
    # ------------------------------------------------------------------

    def get_gene_list(self, dataset_id: str) -> list[str]:
        """Return all gene symbols indexed for a dataset."""
        col_name = self._gene_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
        except Exception:
            return []

        # Retrieve all IDs and extract gene symbols from metadata
        all_data = collection.get(include=["metadatas"])
        return sorted(m["gene_symbol"] for m in all_data["metadatas"])

    def get_celltypes(self, dataset_id: str) -> list[str]:
        """Return all cell types indexed for a dataset."""
        col_name = self._celltype_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
        except Exception:
            return []

        all_data = collection.get(include=["metadatas"])
        return sorted(m["celltype"] for m in all_data["metadatas"])

    def is_indexed(self, dataset_id: str) -> bool:
        """Check if a dataset has been indexed in ChromaDB."""
        col_name = self._gene_collection_name(dataset_id)
        try:
            collection = self.client.get_collection(col_name)
            return collection.count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_dataset(self, dataset_id: str):
        """Remove all collections for a dataset (for re-ingestion)."""
        for name in [
            self._gene_collection_name(dataset_id),
            self._celltype_collection_name(dataset_id),
        ]:
            try:
                self.client.delete_collection(name=name)
                logger.info(f"Deleted collection '{name}'")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
