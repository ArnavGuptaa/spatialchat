#!/usr/bin/env python3
"""
One-time migration script: populate ChromaDB from existing h5ad datasets.

Reads every dataset listed in data/catalog.json, loads the h5ad file,
computes per-gene expression statistics, and indexes everything in ChromaDB.
Run this once after adding ChromaDB to an existing SpatialChat installation.

Usage:
    python scripts/build_vector_index.py            # index all datasets
    python scripts/build_vector_index.py --dataset-id mouse_brain_seqfish  # index one
    python scripts/build_vector_index.py --force     # re-index even if already done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index for SpatialChat")
    parser.add_argument("--dataset-id", default=None,
                        help="Index a single dataset (default: all datasets in catalog)")
    parser.add_argument("--force", action="store_true",
                        help="Re-index even if the dataset is already indexed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from data.vector_store import get_vector_store

    catalog_path = project_root / "data" / "catalog.json"
    if not catalog_path.exists():
        print("ERROR: data/catalog.json not found. Run ingest_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    with open(catalog_path) as f:
        catalog = json.load(f)

    vs = get_vector_store()

    # Determine which datasets to index
    if args.dataset_id:
        if args.dataset_id not in catalog:
            print(f"ERROR: '{args.dataset_id}' not found in catalog.", file=sys.stderr)
            print(f"  Available: {list(catalog.keys())}")
            sys.exit(1)
        dataset_ids = [args.dataset_id]
    else:
        dataset_ids = list(catalog.keys())

    print(f"Building vector index for {len(dataset_ids)} dataset(s)...\n")

    for dataset_id in dataset_ids:
        info = catalog[dataset_id]
        data_file = info.get("data_file", "")
        h5ad_path = project_root / "data" / data_file

        if not h5ad_path.exists():
            print(f"SKIP {dataset_id}: h5ad file not found at {h5ad_path}")
            continue

        if vs.is_indexed(dataset_id) and not args.force:
            n_genes = len(vs.get_gene_list(dataset_id))
            print(f"SKIP {dataset_id}: already indexed ({n_genes} genes). Use --force to re-index.")
            continue

        print(f"Indexing '{dataset_id}'...")
        print(f"  Loading {h5ad_path}...")
        t0 = time.time()

        try:
            adata = ad.read_h5ad(h5ad_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        celltype_col = info.get("celltype", None)
        print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if celltype_col:
            print(f"  Celltype column: {celltype_col}")

        # Delete existing collections if re-indexing
        if args.force:
            vs.delete_dataset(dataset_id)

        try:
            result = vs.index_dataset(dataset_id, adata, celltype_col)
            elapsed = time.time() - t0
            print(f"  Indexed {result['n_genes_indexed']} genes, "
                  f"{result['n_celltypes_indexed']} celltypes "
                  f"in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR indexing: {e}")
            continue

    print("\nDone! Vector index is ready for RAG queries.")


if __name__ == "__main__":
    main()
