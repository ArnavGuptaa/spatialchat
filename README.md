# SpatialChat

A natural language interface for spatial transcriptomics analysis, built on a **supervisor-routed multi-agent architecture** using [LangGraph](https://github.com/langchain-ai/langgraph). The frontend is a [Streamlit](https://streamlit.io/) web app, and all agent runs can be traced and debugged through [LangSmith](https://smith.langchain.com).

Ask questions like *"Show Snap25 expression in the mouse brain Visium dataset"* and get back spatial plots, statistics, and interpretations through conversation.

<p align="center">
  <img src="docs/example_query.png" alt="Example query showing Snap25 spatial expression in mouse brain" width="600">
</p>

## Why This Exists

SpatialChat is designed as a **reference architecture** for building multi-agent systems with LangGraph. The spatial transcriptomics domain is just one application. You can plug in your own sub-agents, tools, and data sources to adapt it to any domain.

The core pattern (supervisor routing to specialized sub-agents, each with their own tools and prompts) is general-purpose and extensible. Fork it, swap out the biology, and build your own agent.

## Quick Start

**1. Install with uv**

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```
OPENAI_API_KEY=sk-...          # or ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=lsv2_...     # optional, for LangSmith tracing
```

**3. Add data**

```bash
# Download the mouse brain seqFISH dataset (~31 MB)
uv run python -c "import squidpy as sq; adata = sq.datasets.seqfish(); adata.write_h5ad('data/anndata/seqfish.h5ad')"

# Download the mouse brain Visium dataset (~314 MB)
uv run python -c "import squidpy as sq; adata = sq.datasets.visium_hne_adata(); adata.write_h5ad('data/anndata/visium.h5ad')"
```

Or bring your own h5ad files (see the [Data Ingestion Guide](docs/ARCHITECTURE.md#data-ingestion-guide)).

**4. Run**

```bash
uv run run-app
```

Or directly with streamlit:

```bash
uv run streamlit run app.py
```

LangGraph Studio (for development and debugging):

```bash
langgraph dev
```

## Testing

The test suite covers tool logic, data caching, catalog loading, routing, and full graph integration. Tests use synthetic AnnData fixtures so you don't need real data for unit tests.

```bash
# Run all unit tests (no API key needed)
uv run pytest tests/test_tools.py -v

# Run integration tests (requires an LLM API key in .env)
uv run pytest tests/test_graph.py -v

# Run everything
uv run pytest -v
```

Adding tests for new tools is straightforward: create a synthetic AnnData fixture, mock the cache, and assert on the JSON output. See `tests/test_tools.py` for examples covering PlotStore behavior, tool result formatting, gene validation (exact, case-insensitive, fuzzy), expression tool outputs, LRU cache eviction, and routing logic.

## Documentation

The documentation is split into focused guides:

- **[Architecture](docs/ARCHITECTURE.md)** covers the graph structure, design decisions, data layer, configuration, LangSmith tracing setup, and project layout.
- **[Extending SpatialChat](docs/EXTENDING.md)** is a step-by-step guide for adding new tools, creating new sub-agents, and wiring them into the graph.

## Included Datasets

| Dataset ID | Description | Cells | Genes | Technology |
|---|---|---|---|---|
| `mouse_brain_seqfish` | Mouse brain sub-ventricular zone | 19,416 | 351 | seqFISH |
| `mouse_brain_visium` | Mouse brain sagittal section | 2,688 | 18,078 | 10x Visium |

## License

MIT
