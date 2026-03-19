from langchain_core.tools import tool

@tool
def calculate_kpi(brand_name: str, metric: str) -> str:
    """Mock business function to calculate KPI for a given brand."""
    # Insert real logic here
    return f"{metric} for {brand_name} is performing at 110% of target."

@tool
def retrieve_brand_data(namespace: str, query: str) -> str:
    """Mock search function demonstrating FAISS retrieval with private namespaces."""
    # In a real FAISS implementation, you'd use the namespace to isolate data access:
    # 1. Option A: Different FAISS indices loaded per namespace (e.g., indices[namespace].similarity_search)
    # 2. Option B: Metadata filtering (e.g., faiss_store.similarity_search(query, filter={"namespace": namespace}))
    return f"Retrieved data for '{query}' completely isolated to brand namespace '{namespace}'"
