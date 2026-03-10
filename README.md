# SpatialChat

Natural language interface for spatial transcriptomics analysis, powered by LangGraph multi-agent architecture.

Ask questions like *"Show Sox2 expression in the mouse brain seqFISH dataset"* and get back spatial plots, statistics, and interpretations — all through conversation.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=lsv2_...     # optional, for LangSmith tracing
```

### 3. Add data

SpatialChat reads spatial transcriptomics data from h5ad files. To get started with the bundled datasets:

```bash
# Download the mouse brain seqFISH dataset (~31 MB)
python -c "import squidpy as sq; adata = sq.datasets.seqfish(); adata.write_h5ad('data/anndata/seqfish.h5ad')"

# Download the mouse brain Visium dataset (~314 MB)
python -c "import squidpy as sq; adata = sq.datasets.visium_hne_adata(); adata.write_h5ad('data/anndata/visium.h5ad')"
```

Or ingest your own data (see [Data Ingestion Guide](#data-ingestion-guide) below).

### 4. Run

**Option A: LangGraph Studio** (recommended for development)

```bash
langgraph dev
```

Open the Studio UI at the URL printed in the terminal.

**Option B: Streamlit** (standalone web app)

```bash
streamlit run app.py
```

---

## Architecture

SpatialChat uses a **supervisor-routed multi-agent** pattern built on LangGraph:

```
User Query
    │
    ▼
┌──────────┐
│  Reset   │  ← clears per-turn state
└────┬─────┘
     ▼
┌──────────────┐
│  Supervisor  │  ← routes to the right sub-agent
└──────┬───────┘
       │ (conditional edges)
       ├──► dataset_finder  ── search, load, validate genes
       ├──► exploratory     ── expression plots, celltype analysis
       ├──► spatial_stats   ── Moran's I, co-occurrence
       └──► neighborhood    ── enrichment, interaction matrices
              │
              ▼
       ┌──────────────┐
       │ Synthesizer  │  ← combines results + suggests follow-ups
       └──────────────┘
              │
              ▼
         Final Answer (text + plots)
```

**Key design decisions:**

- **PlotStore pattern**: Plots are stored by short ID, never as base64 in LLM context. The synthesizer embeds them in the final message.
- **Compact tool outputs**: Tool results are capped at 300 chars. No raw expression arrays or large data in context.
- **Reset node**: Per-invocation state (tool_summaries, plot_ids, visited_agents) is cleared each turn to prevent accumulation.
- **Universal h5ad loader**: All datasets load from h5ad files referenced in `data/catalog.json`.
- **Metadata store**: Pre-computed gene lists and cell types per dataset in `data/metadata/` for fast lookups and fuzzy gene matching.

---

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

This will:

1. **Validate** the h5ad file (checks for `spatial` in obsm, verifies celltype column)
2. **Copy** it to `data/anndata/my_experiment.h5ad`
3. **Update** `data/catalog.json` with the new entry
4. **Build metadata DB** in `data/metadata/my_experiment.json` (gene list + cell types for fast lookup and fuzzy matching)

### Requirements for h5ad files

Your AnnData object must have:

- **`adata.obsm["spatial"]`** — a numpy array of shape (n_cells, 2) with spatial coordinates
- **`adata.obs[celltype_col]`** — (optional but recommended) a column with cell type labels
- **`adata.X`** — gene expression matrix (dense or sparse)
- **`adata.var_names`** — gene symbols

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

The `celltype` field maps to the obs column that the celltype tools will use. The `data_file` path is relative to the `data/` directory.

After manual entry, build the metadata DB:

```python
import anndata as ad
from data.metadata_store import build_metadata_from_adata, save_metadata

adata = ad.read_h5ad("data/anndata/my_dataset.h5ad")
meta = build_metadata_from_adata("my_dataset", adata, "cell_type")
save_metadata("my_dataset", meta)
```

### Metadata store

Each ingested dataset gets a JSON metadata file in `data/metadata/{dataset_id}.json`:

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

This enables:
- **Fuzzy gene matching**: When a user asks for a gene that doesn't exist, `find_similar_genes()` uses 4-tier matching (exact case-insensitive → edit-distance fuzzy → substring → prefix) to suggest alternatives
- **Fast lookups**: Gene and cell type lists are available without loading the full AnnData into memory
- **Tool robustness**: All tools use the metadata store to provide helpful error messages with suggestions

---

## Extending SpatialChat

This section covers how to add new tools, create new sub-agents, and update existing agents.

### Adding a New Tool to an Existing Sub-Agent

This is the simplest extension. For example, adding a new expression analysis tool to the `exploratory` sub-agent:

**Step 1: Define the tool in the appropriate `tools/` file**

Open `tools/expression_tools.py` and add:

```python
class MyNewToolArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Gene symbol (case-sensitive — call validate_gene first)")
    # Add more parameters as needed

@tool(args_schema=MyNewToolArgs)
def my_new_expression_tool(dataset_id: str, gene: str) -> str:
    """Clear description of what this tool does — the LLM reads this."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(
            success=False,
            message="Dataset not loaded.",
            error="Call load_and_summarize_dataset first. "
                  "This tool can [describe what it does] for any loaded dataset."
        )
    if gene not in adata.var_names:
        suggestions = find_similar_genes(dataset_id, gene, n=5)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        return tool_result(
            success=False,
            message=f"Gene '{gene}' not in dataset.{hint}",
            error=f"Call validate_gene first. This tool can [describe capability]."
        )

    # Your analysis logic here
    # ...

    # If generating a plot:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ... plot ...
    pid = fig_to_plot_id(fig)  # Stores in PlotStore, returns short ID

    return tool_result(
        success=True,
        message=f"Short summary of results",
        data={"key": "value"},  # Compact data only
        plot_id=pid,            # Optional: reference to stored plot
    )
```

**Key conventions for tools:**
- Always use `tool_result()` for return values — it formats JSON consistently
- On failure, include a description of what the tool CAN do in the `error` field
- Use `find_similar_genes()` from `data.metadata_store` when a gene isn't found
- Never return large data (arrays, full gene lists) — only summary statistics
- Use `fig_to_plot_id(fig)` for plots — never base64 in results

**Step 2: Register the tool in the TOOLS list**

At the bottom of the same file, add to the registry:

```python
EXPRESSION_TOOLS = [
    get_gene_expression_spatial,
    show_spatial_domains,
    compare_expression,
    plot_celltype_spatial,
    gene_expression_by_celltype,
    my_new_expression_tool,  # ← add here
]
```

**Step 3: Update the sub-agent prompt**

In `agents/sub_agents.py`, update the `EXPLORATORY_PROMPT` to mention your new tool:

```python
EXPLORATORY_PROMPT = (
    "You analyze gene expression and cell types in a loaded dataset.\n"
    "Available tools:\n"
    "- get_gene_expression_spatial: plot a gene on spatial coords\n"
    "- show_spatial_domains: plot any annotation spatially\n"
    "- compare_expression: compare gene between two groups\n"
    "- plot_celltype_spatial: plot cell types on spatial coords\n"
    "- gene_expression_by_celltype: bar chart of gene expression per cell type\n"
    "- my_new_expression_tool: [brief description]\n"  # ← add here
    "Use dataset_id from context. Call only the tool needed.\n"
    "Keep response short — key stats and interpretation only."
)
```

That's it! The tool is now available. The LLM reads the prompt + tool docstring to decide when to use it.

### Creating a New Sub-Agent

For entirely new analysis categories (e.g., trajectory analysis, gene regulatory networks):

**Step 1: Create tools**

Create a new file `tools/my_analysis_tools.py`:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tools.base import fig_to_plot_id, tool_result
from data.loaders import get_cache
from data.metadata_store import find_similar_genes

class MyToolArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    param: str = Field(description="Analysis parameter")

@tool(args_schema=MyToolArgs)
def my_analysis_tool(dataset_id: str, param: str) -> str:
    """Run my custom analysis on the dataset."""
    adata = get_cache().get(dataset_id)
    if adata is None:
        return tool_result(
            success=False,
            message="Dataset not loaded.",
            error="Call load_and_summarize_dataset first. "
                  "This tool performs [description]. "
                  "It can also [list other capabilities]."
        )

    # Your analysis here...
    result_value = 42

    return tool_result(
        success=True,
        message=f"Analysis result: {result_value}",
        data={"result": result_value},
    )

MY_ANALYSIS_TOOLS = [my_analysis_tool]
```

**Step 2: Create the sub-agent factory**

In `agents/sub_agents.py`, add the import and factory:

```python
from tools.my_analysis_tools import MY_ANALYSIS_TOOLS

MY_ANALYSIS_PROMPT = (
    "You run custom analyses on a loaded dataset.\n"
    "Available tools:\n"
    "- my_analysis_tool: [description]\n"
    "Use dataset_id from context. Be concise."
)

def create_my_analysis_agent():
    llm = get_settings().get_sub_agent_llm()
    return build_sub_agent(llm, MY_ANALYSIS_TOOLS, MY_ANALYSIS_PROMPT)
```

**Step 3: Add agent name constant to state**

In `agents/state.py`:

```python
# Agent name constants
DATASET_FINDER = "dataset_finder"
EXPLORATORY = "exploratory"
SPATIAL_STATS = "spatial_stats"
NEIGHBORHOOD = "neighborhood"
MY_ANALYSIS = "my_analysis"      # ← add here
FINISH = "FINISH"
```

**Step 4: Register in the graph**

In `graph.py`, update imports and add the node:

```python
from agents.sub_agents import (
    create_dataset_finder_agent, create_exploratory_agent,
    create_spatial_stats_agent, create_neighborhood_agent,
    create_my_analysis_agent,  # ← add import
)
from agents.state import (
    SpatialChatState, DATASET_FINDER, EXPLORATORY,
    SPATIAL_STATS, NEIGHBORHOOD, MY_ANALYSIS, FINISH,  # ← add MY_ANALYSIS
)

# In build_graph():
g.add_node(MY_ANALYSIS, _make_node(MY_ANALYSIS, create_my_analysis_agent))

# Update conditional edges:
g.add_conditional_edges("supervisor", route_from_supervisor, {
    DATASET_FINDER: DATASET_FINDER,
    EXPLORATORY: EXPLORATORY,
    SPATIAL_STATS: SPATIAL_STATS,
    NEIGHBORHOOD: NEIGHBORHOOD,
    MY_ANALYSIS: MY_ANALYSIS,      # ← add here
    "synthesizer": "synthesizer",
})

# Add return edge:
for a in [DATASET_FINDER, EXPLORATORY, SPATIAL_STATS, NEIGHBORHOOD, MY_ANALYSIS]:
    g.add_edge(a, "supervisor")

# Update route_from_supervisor:
def route_from_supervisor(state: SpatialChatState) -> str:
    nxt = state.get("next_agent")
    valid = {DATASET_FINDER, EXPLORATORY, SPATIAL_STATS, NEIGHBORHOOD, MY_ANALYSIS}
    return nxt if nxt in valid else "synthesizer"
```

**Step 5: Update the supervisor prompt**

In `agents/supervisor.py`:

```python
SUPERVISOR_PROMPT = """You route user queries to specialist sub-agents. Pick ONE at a time.

AGENTS:
- dataset_finder: Search/load datasets. Call FIRST if no dataset loaded.
- exploratory: Gene expression plots. Requires dataset.
- spatial_stats: Moran's I, co-occurrence. Requires dataset.
- neighborhood: Cell neighbors, interactions. Requires dataset.
- my_analysis: [Brief description]. Requires dataset.

RULES:
- Dataset loaded? Skip dataset_finder.
- Each agent: call AT MOST ONCE per turn.
- Have enough results? → FINISH.
- NEVER re-route to the same agent."""
```

Update `SupervisorDecision`:

```python
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Brief routing rationale")
    next_agent: Literal[
        "dataset_finder", "exploratory", "spatial_stats",
        "neighborhood", "my_analysis", "FINISH"
    ]
```

### Updating an Existing Sub-Agent

To modify an existing sub-agent's behavior:

1. **Change tool behavior**: Edit the tool function in `tools/`. The tool's docstring and error messages guide the LLM.
2. **Change routing logic**: Edit `SUPERVISOR_PROMPT` in `agents/supervisor.py` to change when the agent is called.
3. **Change sub-agent reasoning**: Edit the prompt in `agents/sub_agents.py` (e.g., `EXPLORATORY_PROMPT`).
4. **Add/remove tools**: Update the `TOOLS` list in the tools file and the prompt in `sub_agents.py`.

---

## Available Tools

| Tool | Sub-Agent | Description |
|------|-----------|-------------|
| `search_datasets` | dataset_finder | Search catalog by tissue, species, technology |
| `load_and_summarize_dataset` | dataset_finder | Load h5ad into memory |
| `validate_gene` | dataset_finder | Check if a gene exists (fuzzy matching with suggestions) |
| `get_gene_expression_spatial` | exploratory | Plot gene expression on spatial coords |
| `show_spatial_domains` | exploratory | Plot any annotation column spatially |
| `compare_expression` | exploratory | Mann-Whitney U test between two groups |
| `plot_celltype_spatial` | exploratory | Plot cell types on spatial coords (auto-resolves column) |
| `gene_expression_by_celltype` | exploratory | Bar chart of gene expression per cell type |
| `spatial_autocorrelation` | spatial_stats | Moran's I test for spatial autocorrelation |
| `co_occurrence` | spatial_stats | Cell type co-occurrence analysis |
| `neighborhood_enrichment` | neighborhood | Neighborhood enrichment analysis |
| `interaction_matrix` | neighborhood | Cell-cell interaction matrix |

---

## Included Datasets

| Dataset ID | Description | Cells | Genes | Technology |
|------------|-------------|-------|-------|------------|
| `mouse_brain_seqfish` | Mouse brain sub-ventricular zone | 19,416 | 351 | seqFISH |
| `mouse_brain_visium` | Mouse brain sagittal section | 2,688 | 18,078 | 10x Visium |

---

## Configuration

All settings are in `.env`. See `.env.example` for the full list.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `LLM_PROVIDER` | Force provider (`openai` or `anthropic`) | auto-detect |
| `LLM_MODEL` | Model name | `gpt-4o` / `claude-sonnet-4-20250514` |
| `SUB_AGENT_MODEL` | Cheaper model for sub-agents | same as main |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing | — |
| `LANGCHAIN_PROJECT` | LangSmith project name | `spatialchat` |
| `MAX_LOADED_DATASETS` | Max datasets in memory cache | `3` |

---

## LangSmith Tracing

If `LANGCHAIN_API_KEY` is set, all LangGraph runs are traced to LangSmith. You can view:

- Supervisor routing decisions
- Sub-agent tool calls and responses
- Token usage per step
- Full message history

Visit [smith.langchain.com](https://smith.langchain.com) and navigate to your project.

---

## Project Structure

```
spatialchat/
├── graph.py                 # Main LangGraph wiring
├── app.py                   # Streamlit frontend
├── langgraph.json           # LangGraph Studio config
├── agents/
│   ├── state.py             # Shared state schema
│   ├── supervisor.py        # Supervisor + synthesizer nodes
│   └── sub_agents.py        # Sub-agent factories (bind_tools)
├── tools/
│   ├── base.py              # PlotStore + tool_result helper
│   ├── dataset_tools.py     # search, load, validate (fuzzy gene matching)
│   ├── expression_tools.py  # spatial plots, celltype analysis
│   ├── stats_tools.py       # Moran's I, co-occurrence
│   └── neighbor_tools.py    # enrichment, interaction matrix
├── data/
│   ├── catalog.json         # Dataset catalog
│   ├── loaders.py           # Universal h5ad loader + cache
│   ├── metadata_store.py    # Pre-computed gene lists + celltypes per dataset
│   ├── metadata/            # JSON metadata files per dataset
│   └── anndata/             # h5ad files (gitignored)
├── config/
│   └── settings.py          # Pydantic settings from .env
├── scripts/
│   └── ingest_dataset.py    # CLI for adding new datasets + metadata
└── tests/
    ├── test_tools.py        # Unit tests for tools
    └── test_graph.py        # Graph routing tests
```
