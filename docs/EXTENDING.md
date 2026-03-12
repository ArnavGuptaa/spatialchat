# Extending SpatialChat

This guide covers how to add new tools, create new sub-agents, and wire them into the graph. SpatialChat is designed as an extensible architecture: the spatial transcriptomics domain is just one application of the supervisor-routed multi-agent pattern.

## Adding a New Tool to an Existing Sub-Agent

This is the simplest extension. For example, adding a new expression analysis tool to the `exploratory` sub-agent.

**Step 1: Define the tool in the appropriate `tools/` file**

Open `tools/expression_tools.py` and add:

```python
class MyNewToolArgs(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    gene: str = Field(description="Gene symbol (case-sensitive, call validate_gene first)")

@tool(args_schema=MyNewToolArgs)
def my_new_expression_tool(dataset_id: str, gene: str) -> str:
    """Clear description of what this tool does. The LLM reads this."""
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

Key conventions for tools:

- Always use `tool_result()` for return values. It formats JSON consistently.
- On failure, include a description of what the tool CAN do in the `error` field. This helps the LLM recover.
- Use `find_similar_genes()` from `data.metadata_store` when a gene isn't found.
- Use `search_genes_semantic()` or `find_expression_similar_genes()` from `data.metadata_store` for RAG-style gene discovery.
- Never return large data (arrays, full gene lists). Only summary statistics.
- Use `fig_to_plot_id(fig)` for plots. Never base64 in results.

**Step 2: Register the tool in the TOOLS list**

At the bottom of the same file:

```python
EXPRESSION_TOOLS = [
    get_gene_expression_spatial,
    show_spatial_domains,
    compare_expression,
    plot_celltype_spatial,
    gene_expression_by_celltype,
    my_new_expression_tool,  # add here
]
```

**Step 3: Update the sub-agent prompt**

In `agents/sub_agents.py`, update `EXPLORATORY_PROMPT` to mention your new tool:

```python
EXPLORATORY_PROMPT = (
    "You analyze gene expression and cell types in a loaded dataset.\n"
    "Available tools:\n"
    "- get_gene_expression_spatial: plot a gene on spatial coords\n"
    "- show_spatial_domains: plot any annotation spatially\n"
    "- compare_expression: compare gene between two groups\n"
    "- plot_celltype_spatial: plot cell types on spatial coords\n"
    "- gene_expression_by_celltype: bar chart of gene expression per cell type\n"
    "- my_new_expression_tool: [brief description]\n"  # add here
    "Use dataset_id from context. Call only the tool needed.\n"
    "Keep response short, key stats and interpretation only."
)
```

That's it. The tool is now available. The LLM reads the prompt and tool docstring to decide when to use it.

**Step 4: Add a unit test**

In `tests/test_tools.py`, add a test using the existing mock infrastructure:

```python
class TestMyNewTool:
    def test_returns_plot_id(self, mock_cache):
        from tools.expression_tools import my_new_expression_tool
        from tools.base import clear_plot_store, get_plot_base64

        clear_plot_store()
        with patch("tools.expression_tools.get_cache", return_value=mock_cache):
            result = json.loads(my_new_expression_tool.invoke({
                "dataset_id": "test", "gene": "Snap25"
            }))

        assert result["success"] is True
        assert "plot_id" in result
        assert "plot_base64" not in result
        assert get_plot_base64(result["plot_id"]) is not None

    def test_missing_dataset(self, mock_cache):
        from tools.expression_tools import my_new_expression_tool

        mock_cache.get.return_value = None
        with patch("tools.expression_tools.get_cache", return_value=mock_cache):
            result = json.loads(my_new_expression_tool.invoke({
                "dataset_id": "missing", "gene": "Snap25"
            }))
        assert result["success"] is False
```

The test suite uses synthetic AnnData fixtures and mocked caches, so no real data or API keys are needed for unit tests. Run with `pytest tests/test_tools.py -v`.

## Creating a New Sub-Agent

For entirely new analysis categories (e.g., trajectory analysis, gene regulatory networks).

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

    result_value = 42  # your analysis here

    return tool_result(
        success=True,
        message=f"Analysis result: {result_value}",
        data={"result": result_value},
    )

MY_ANALYSIS_TOOLS = [my_analysis_tool]
```

**Step 2: Create the sub-agent factory**

In `agents/sub_agents.py`:

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
DATASET_FINDER = "dataset_finder"
EXPLORATORY = "exploratory"
SPATIAL_STATS = "spatial_stats"
NEIGHBORHOOD = "neighborhood"
MY_ANALYSIS = "my_analysis"      # add here
FINISH = "FINISH"
```

**Step 4: Register in the graph**

In `graph.py`, update imports and add the node:

```python
from agents.sub_agents import (
    create_dataset_finder_agent, create_exploratory_agent,
    create_spatial_stats_agent, create_neighborhood_agent,
    create_my_analysis_agent,  # add import
)
from agents.state import (
    SpatialChatState, DATASET_FINDER, EXPLORATORY,
    SPATIAL_STATS, NEIGHBORHOOD, MY_ANALYSIS, FINISH,  # add MY_ANALYSIS
)

# In build_graph():
g.add_node(MY_ANALYSIS, _make_node(MY_ANALYSIS, create_my_analysis_agent))

# Update conditional edges:
g.add_conditional_edges("supervisor", route_from_supervisor, {
    DATASET_FINDER: DATASET_FINDER,
    EXPLORATORY: EXPLORATORY,
    SPATIAL_STATS: SPATIAL_STATS,
    NEIGHBORHOOD: NEIGHBORHOOD,
    MY_ANALYSIS: MY_ANALYSIS,      # add here
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
- Have enough results? -> FINISH.
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

**Step 6: Add tests**

Add a routing test in `tests/test_tools.py`:

```python
class TestMyAnalysisRouting:
    def test_route_to_my_analysis(self):
        from graph import route_from_supervisor
        assert route_from_supervisor({"next_agent": "my_analysis"}) == "my_analysis"
```

And a tool test following the same mock pattern shown in Step 4 of the tool section above.

## Updating an Existing Sub-Agent

To modify an existing sub-agent's behavior:

1. **Change tool behavior**: Edit the tool function in `tools/`. The tool's docstring and error messages guide the LLM.
2. **Change routing logic**: Edit `SUPERVISOR_PROMPT` in `agents/supervisor.py` to change when the agent is called.
3. **Change sub-agent reasoning**: Edit the prompt in `agents/sub_agents.py` (e.g., `EXPLORATORY_PROMPT`).
4. **Add/remove tools**: Update the `TOOLS` list in the tools file and the prompt in `sub_agents.py`.

## Adapting to a Different Domain

The architecture is domain-agnostic. To adapt SpatialChat for a completely different use case:

1. Replace the tools in `tools/` with your domain-specific logic.
2. Update `data/` with your data loading and catalog system.
3. Rewrite the sub-agent prompts in `agents/sub_agents.py`.
4. Update the supervisor prompt in `agents/supervisor.py` with your new agent names and routing rules.
5. Keep the graph wiring in `graph.py` mostly as-is. The supervisor/sub-agent/synthesizer pattern works across domains.

The Streamlit frontend (`app.py`) is generic and should work without changes for any text-and-plot conversational agent.
