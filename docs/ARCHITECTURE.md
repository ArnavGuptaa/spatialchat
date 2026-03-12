# Architecture

SpatialChat uses a **supervisor-routed multi-agent** pattern built on LangGraph. A Streamlit frontend handles the conversational UI, and LangSmith provides full observability into every LLM call, tool invocation, and routing decision.

## Graph Structure

```
User Query
    |
    v
+----------+
|  Reset   |   clears per-turn state
+----+-----+
     v
+--------------+
|  Supervisor  |   routes to the right sub-agent
+------+-------+
       | (conditional edges)
       +---> dataset_finder    search, load, validate genes
       +---> exploratory       expression plots, celltype analysis
       +---> spatial_stats     Moran's I, co-occurrence
       +---> neighborhood      enrichment, interaction matrices
              |
              v
       +--------------+
       | Synthesizer  |   combines results + suggests follow-ups
       +--------------+
              |
              v
         Final Answer (text + plots)
```

Each sub-agent loops internally using LangChain's `bind_tools` ReAct pattern (LLM call, tool call, tool result, repeat until done), then returns a compact summary to the supervisor. The supervisor decides what to do next: route to another sub-agent or finish.

## Design Decisions

**PlotStore pattern.** Plots are stored by short ID, never as base64 in LLM context. Tools call `fig_to_plot_id(fig)` which saves the rendered PNG and returns an 8-character hex ID. The synthesizer embeds plots in the final message using these IDs.

**Compact tool outputs.** Tool results are capped at 300 characters. No raw expression arrays or large data ever enters the LLM context. Tools return summary statistics only.

**Reset node.** Per-invocation state (`tool_summaries`, `plot_ids`, `visited_agents`) is cleared at the start of each turn. This prevents accumulation across multi-turn conversations while preserving `messages` and `active_dataset_id`.

**Universal h5ad loader.** All datasets load from h5ad files referenced in `data/catalog.json`. No per-dataset loader functions.

**Metadata store with ChromaDB vector backend.** Gene expression statistics and cell type metadata are indexed in ChromaDB as numeric embeddings, enabling RAG-style semantic search ("find markers of endothelium") and vector similarity search ("find genes with similar expression patterns to T"). JSON metadata files in `data/metadata/` are kept as a lightweight fallback. The embedding strategy uses per-gene summary statistics (mean, median, pct_expressing, std, per-celltype means) normalised to unit vectors, so cosine distance captures expression profile shape rather than magnitude.

## Data Ingestion Guide

### Using the ingestion script

The easiest way to add a new dataset:

```bash
python scripts/ingest_dataset.py \
    --h5ad /path/to/my_data.h5ad \
    --dataset-id my_experiment \
    --name "My Spatial Experiment" \
    --species mouse \
    --tissue brain \
    --technology MERFISH \
    --celltype-col cell_type \
    --description "MERFISH data from mouse cortex"
```

This will validate the h5ad file (checks for `spatial` in obsm, verifies the celltype column), copy it to `data/anndata/my_experiment.h5ad`, update `data/catalog.json`, build a metadata JSON in `data/metadata/`, and index gene expression embeddings + cell type metadata in ChromaDB (`data/vectordb/`).

### Requirements for h5ad files

Your AnnData object must have:

- `adata.obsm["spatial"]` with shape (n_cells, 2) for spatial coordinates
- `adata.obs[celltype_col]` (optional but recommended) for cell type labels
- `adata.X` for the gene expression matrix (dense or sparse)
- `adata.var_names` for gene symbols

### Manual catalog entry

You can also edit `data/catalog.json` directly:

```json
{
  "my_dataset": {
    "display_name": "My Dataset",
    "tissue": "brain",
    "region": "cortex",
    "species": "mouse",
    "technology": "MERFISH",
    "source": "user-provided",
    "description": "...",
    "n_spots_approx": 50000,
    "n_genes_approx": 500,
    "annotations": ["cell_type", "region"],
    "celltype": "cell_type",
    "data_file": "anndata/my_dataset.h5ad",
    "reference": "",
    "license": ""
  }
}
```

The `celltype` field maps to the obs column that celltype tools will use. The `data_file` path is relative to the `data/` directory.

After manual entry, build the metadata:

```python
import anndata as ad
from data.metadata_store import build_metadata_from_adata, save_metadata

adata = ad.read_h5ad("data/anndata/my_dataset.h5ad")
meta = build_metadata_from_adata("my_dataset", adata, "cell_type")
save_metadata("my_dataset", meta)
```

### Metadata store

Each ingested dataset gets a JSON metadata file in `data/metadata/{dataset_id}.json` plus ChromaDB collections in `data/vectordb/`:

```json
{
  "dataset_id": "mouse_brain_seqfish",
  "genes": ["Abcc4", "Acp5", "..."],
  "celltypes": ["Astrocytes", "Neurons", "..."],
  "n_cells": 19416,
  "n_genes": 351,
  "annotations": ["celltype_mapped_refined"],
  "celltype_column": "celltype_mapped_refined"
}
```

The ChromaDB vector store (`data/vector_store.py`) indexes two collections per dataset: `gene_metadata_{dataset_id}` stores per-gene expression statistics as numeric embeddings for similarity search, and `celltype_metadata_{dataset_id}` stores cell type info with marker genes for keyword search. Gene lookups try ChromaDB first, then fall back to JSON if the vector store is unavailable.

To rebuild the vector index for existing datasets:

```bash
python scripts/build_vector_index.py           # index all datasets
python scripts/build_vector_index.py --force    # re-index even if already done
```

## Available Tools

| Tool | Sub-Agent | Description |
|---|---|---|
| `search_datasets` | dataset_finder | Search catalog by tissue, species, technology |
| `load_and_summarize_dataset` | dataset_finder | Load h5ad into memory |
| `validate_gene` | dataset_finder | Check if a gene exists (fuzzy matching with suggestions) |
| `get_gene_expression_spatial` | exploratory | Plot gene expression on spatial coords |
| `show_spatial_domains` | exploratory | Plot any annotation column spatially |
| `compare_expression` | exploratory | Mann-Whitney U test between two groups |
| `plot_celltype_spatial` | exploratory | Plot cell types on spatial coords (auto-resolves column) |
| `gene_expression_by_celltype` | exploratory | Bar chart of gene expression per cell type |
| `rag_query_genes` | exploratory | RAG: search genes by biological description |
| `rag_find_similar_genes` | exploratory | RAG: find genes with similar expression profiles (vector similarity) |
| `rag_query_celltypes` | exploratory | RAG: search cell types and their marker genes |
| `spatial_autocorrelation` | spatial_stats | Moran's I test for spatial autocorrelation |
| `co_occurrence` | spatial_stats | Cell type co-occurrence analysis |
| `neighborhood_enrichment` | neighborhood | Neighborhood enrichment analysis |
| `interaction_matrix` | neighborhood | Cell-cell interaction matrix |

## Configuration

All settings are in `.env`. See `.env.example` for the full list.

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | (none) |
| `ANTHROPIC_API_KEY` | Anthropic API key | (none) |
| `LLM_PROVIDER` | Force provider (`openai` or `anthropic`) | auto-detect |
| `LLM_MODEL` | Model name | `gpt-4o` / `claude-sonnet-4-20250514` |
| `SUB_AGENT_MODEL` | Cheaper model for sub-agents | same as main |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing | (none) |
| `LANGCHAIN_PROJECT` | LangSmith project name | `spatialchat` |
| `MAX_LOADED_DATASETS` | Max datasets in memory cache | `3` |
| `VECTORDB_DIR` | ChromaDB storage directory | `./data/vectordb` |

## LangSmith Tracing

If `LANGCHAIN_API_KEY` is set in your `.env`, all LangGraph runs are automatically traced to [LangSmith](https://smith.langchain.com). You can view supervisor routing decisions, sub-agent tool calls and responses, token usage per step, and the full message history for every conversation turn.

This is the primary debugging tool for the agent graph. When something routes incorrectly or a tool returns unexpected output, the LangSmith trace shows exactly what happened at each step.

## Project Structure

```
spatialchat/
+-- graph.py                 # Main LangGraph wiring
+-- app.py                   # Streamlit frontend
+-- langgraph.json           # LangGraph Studio config
+-- agents/
|   +-- state.py             # Shared state schema
|   +-- supervisor.py        # Supervisor + synthesizer nodes
|   +-- sub_agents.py        # Sub-agent factories (bind_tools)
+-- tools/
|   +-- base.py              # PlotStore + tool_result helper
|   +-- dataset_tools.py     # search, load, validate (fuzzy gene matching)
|   +-- expression_tools.py  # spatial plots, celltype analysis
|   +-- rag_tools.py         # RAG retrieval tools (gene/celltype search via ChromaDB)
|   +-- stats_tools.py       # Moran's I, co-occurrence
|   +-- neighbor_tools.py    # enrichment, interaction matrix
+-- data/
|   +-- catalog.json         # Dataset catalog
|   +-- loaders.py           # Universal h5ad loader + cache
|   +-- metadata_store.py    # Gene/celltype lookups (ChromaDB primary, JSON fallback)
|   +-- vector_store.py      # ChromaDB vector store for gene expression RAG
|   +-- metadata/            # JSON metadata files per dataset (fallback)
|   +-- vectordb/            # ChromaDB persistent storage (gitignored)
|   +-- anndata/             # h5ad files (gitignored)
+-- config/
|   +-- settings.py          # Pydantic settings from .env
+-- scripts/
|   +-- ingest_dataset.py    # CLI for adding new datasets + metadata + vector indexing
|   +-- build_vector_index.py # One-time migration: build ChromaDB index from existing datasets
+-- tests/
    +-- test_tools.py        # Unit tests for tools
    +-- test_graph.py        # Graph routing tests
```
