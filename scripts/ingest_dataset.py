#!/usr/bin/env python3
"""
Data ingestion script for SpatialChat.

Validates an h5ad file and adds it to the dataset catalog so that
SpatialChat can query it via natural language. Also builds a metadata
database (gene list + cell types) for fast lookups without loading
the full AnnData.

Usage — ingest a custom h5ad file:

    python scripts/ingest_dataset.py \
        --h5ad /path/to/my_data.h5ad \
        --dataset-id my_experiment \
        --name "My Spatial Experiment" \
        --species mouse \
        --tissue brain \
        --technology seqFISH \
        --celltype-col cell_type

Usage — ingest built-in Squidpy datasets:

    # First, download the data using Squidpy's dataset loaders.
    # These ship with Squidpy and require no extra accounts or downloads.

    # 1) Mouse brain seqFISH (~31 MB):
    python -c "
    import squidpy as sq
    adata = sq.datasets.seqfish()
    adata.write_h5ad('data/anndata/seqfish.h5ad')
    "
    python scripts/ingest_dataset.py \
        --h5ad data/anndata/seqfish.h5ad \
        --dataset-id mouse_brain_seqfish \
        --name "Mouse Brain Sub-Ventricular Zone (seqFISH)" \
        --species mouse --tissue brain --technology seqFISH \
        --celltype-col celltype_mapped_refined \
        --region "sub-ventricular zone" \
        --description "seqFISH spatial transcriptomics of the mouse brain sub-ventricular zone. Single-cell resolution with targeted gene panel." \
        --reference "Lohoff et al. 2021, Nature Biotechnology" \
        --license "CC BY 4.0" --no-copy

    # 2) Mouse brain 10x Visium (~314 MB):
    python -c "
    import squidpy as sq
    adata = sq.datasets.visium_hne_adata()
    adata.write_h5ad('data/anndata/visium.h5ad')
    "
    python scripts/ingest_dataset.py \
        --h5ad data/anndata/visium.h5ad \
        --dataset-id mouse_brain_visium \
        --name "Mouse Brain Sagittal (10x Visium)" \
        --species mouse --tissue brain --technology "10x Visium" \
        --celltype-col cluster \
        --region "sagittal section" \
        --description "10x Visium spatial transcriptomics of the mouse brain sagittal section. Spot-level resolution with whole-transcriptome coverage." \
        --reference "10x Genomics" \
        --license "CC BY 4.0" --no-copy

    # 3) Mouse brain 10x Visium fluorescent:
    python -c "
    import squidpy as sq
    adata = sq.datasets.visium_fluo_adata()
    adata.write_h5ad('data/anndata/visium_fluo.h5ad')
    "
    python scripts/ingest_dataset.py \
        --h5ad data/anndata/visium_fluo.h5ad \
        --dataset-id mouse_brain_visium_fluo \
        --name "Mouse Brain Visium Fluorescent" \
        --species mouse --tissue brain --technology "10x Visium" \
        --celltype-col cluster \
        --description "10x Visium fluorescent spatial transcriptomics of the mouse brain." \
        --reference "10x Genomics" \
        --license "CC BY 4.0" --no-copy

    # 4) Four-i (iterative indirect immunofluorescence imaging):
    python -c "
    import squidpy as sq
    adata = sq.datasets.four_i()
    adata.write_h5ad('data/anndata/four_i.h5ad')
    "
    python scripts/ingest_dataset.py \
        --h5ad data/anndata/four_i.h5ad \
        --dataset-id four_i \
        --name "4i Protein Imaging" \
        --species human --tissue various --technology 4i \
        --description "Iterative indirect immunofluorescence imaging (4i) dataset." \
        --reference "Squidpy built-in" --no-copy

    # 5) MIBI-TOF (multiplexed ion beam imaging):
    python -c "
    import squidpy as sq
    adata = sq.datasets.mibitof()
    adata.write_h5ad('data/anndata/mibitof.h5ad')
    "
    python scripts/ingest_dataset.py \
        --h5ad data/anndata/mibitof.h5ad \
        --dataset-id mibitof \
        --name "MIBI-TOF Breast Cancer" \
        --species human --tissue breast --technology MIBI-TOF \
        --celltype-col Cluster \
        --description "Multiplexed ion beam imaging dataset of breast cancer tissue." \
        --reference "Squidpy built-in" --no-copy

This will:
  1. Validate the h5ad (spatial coords, celltype column, gene count)
  2. Copy it to data/anndata/{dataset_id}.h5ad
  3. Add an entry to data/catalog.json
  4. Build metadata DB (gene list + celltypes) in data/metadata/{dataset_id}.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad


def main():
    parser = argparse.ArgumentParser(
        description="Ingest an h5ad file into SpatialChat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--h5ad", required=True, help="Path to the h5ad file")
    parser.add_argument("--dataset-id", required=True,
                        help="Unique ID (e.g. 'mouse_cortex_merfish'). Use snake_case.")
    parser.add_argument("--name", required=True,
                        help="Human-readable display name")
    parser.add_argument("--species", required=True,
                        help="Species (e.g. 'mouse', 'human')")
    parser.add_argument("--tissue", required=True,
                        help="Tissue type (e.g. 'brain', 'liver')")
    parser.add_argument("--technology", required=True,
                        help="Technology (e.g. 'seqFISH', '10x Visium', 'MERFISH')")
    parser.add_argument("--celltype-col", default=None,
                        help="Column in obs containing cell type labels (optional)")
    parser.add_argument("--description", default="",
                        help="Short description of the dataset")
    parser.add_argument("--region", default="",
                        help="Anatomical region (e.g. 'cortex', 'hippocampus')")
    parser.add_argument("--reference", default="",
                        help="Publication reference")
    parser.add_argument("--license", default="",
                        help="Data license (e.g. 'CC BY 4.0')")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy h5ad file — use symlink instead")

    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────
    h5ad_path = Path(args.h5ad).resolve()
    if not h5ad_path.exists():
        print(f"ERROR: File not found: {h5ad_path}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    anndata_dir = data_dir / "anndata"
    catalog_path = data_dir / "catalog.json"

    anndata_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate h5ad ────────────────────────────────────────
    print(f"Reading {h5ad_path}...")
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"ERROR: Could not read h5ad file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  Obs columns: {list(adata.obs.columns)}")
    print(f"  Obsm keys: {list(adata.obsm.keys())}")

    # Check spatial coordinates
    if "spatial" not in adata.obsm:
        print("ERROR: No 'spatial' key in adata.obsm. "
              "SpatialChat requires spatial coordinates.", file=sys.stderr)
        print("  Available obsm keys:", list(adata.obsm.keys()))
        sys.exit(1)

    spatial = adata.obsm["spatial"]
    print(f"  Spatial coordinates: shape {spatial.shape}")

    # Check celltype column
    celltype_col = args.celltype_col
    if celltype_col:
        if celltype_col not in adata.obs.columns:
            print(f"ERROR: Celltype column '{celltype_col}' not found in obs.",
                  file=sys.stderr)
            print(f"  Available columns: {list(adata.obs.columns)}")
            sys.exit(1)
        n_types = adata.obs[celltype_col].nunique()
        print(f"  Cell types ({celltype_col}): {n_types} unique types")
    else:
        print("  No celltype column specified (celltype tools won't work for this dataset)")

    # ── Copy or symlink h5ad ─────────────────────────────────
    dest_path = anndata_dir / f"{args.dataset_id}.h5ad"
    relative_data_file = f"anndata/{args.dataset_id}.h5ad"

    if dest_path.exists():
        print(f"WARNING: {dest_path} already exists. Overwriting.")

    if args.no_copy:
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()
        dest_path.symlink_to(h5ad_path)
        print(f"  Symlinked: {dest_path} -> {h5ad_path}")
    else:
        if h5ad_path.resolve() == dest_path.resolve():
            print(f"  File already at destination: {dest_path}")
        else:
            print(f"  Copying to {dest_path}...")
            shutil.copy2(h5ad_path, dest_path)
            print(f"  Copied ({dest_path.stat().st_size / 1e6:.1f} MB)")

    # ── Update catalog.json ──────────────────────────────────
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = json.load(f)
    else:
        catalog = {}

    annotations = list(adata.obs.columns)

    entry = {
        "display_name": args.name,
        "tissue": args.tissue,
        "region": args.region,
        "species": args.species,
        "technology": args.technology,
        "source": "user-provided",
        "description": args.description,
        "n_spots_approx": adata.shape[0],
        "n_genes_approx": adata.shape[1],
        "annotations": annotations[:15],  # Cap at 15 for readability
        "data_file": relative_data_file,
        "reference": args.reference,
        "license": args.license,
    }

    if celltype_col:
        entry["celltype"] = celltype_col

    catalog[args.dataset_id] = entry

    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"  Catalog updated: {catalog_path}")

    # ── Build metadata database ──────────────────────────────
    # Import here to avoid circular imports during standalone usage
    sys.path.insert(0, str(project_root))
    from data.metadata_store import build_metadata_from_adata, save_metadata

    metadata = build_metadata_from_adata(args.dataset_id, adata, celltype_col)
    meta_path = save_metadata(args.dataset_id, metadata)

    print(f"  Metadata DB: {meta_path}")
    print(f"    Genes indexed: {len(metadata['genes'])}")
    print(f"    Cell types indexed: {len(metadata['celltypes'])}")

    # ── Done ─────────────────────────────────────────────────
    print(f"\nSuccessfully ingested '{args.dataset_id}'!")
    print(f"  Data file: {dest_path}")
    print(f"\nYou can now query this dataset in SpatialChat:")
    print(f'  "Load the {args.name} dataset"')
    if celltype_col:
        print(f'  "Show cell types spatially in {args.dataset_id}"')
    print(f'  "What genes are available in {args.dataset_id}?"')


if __name__ == "__main__":
    main()
