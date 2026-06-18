from mcp.server.fastmcp import FastMCP


mcp = FastMCP("product_support")


@mcp.tool()
def lookup_product_manual(product_or_sku: str, question: str) -> str:
    """Look up product manual guidance for a product or SKU."""
    if "e01" in question.casefold():
        return (
            f"For {product_or_sku}, E01 usually means the device needs a restart, "
            "a filter compartment check, and a power adapter check."
        )
    return (
        f"Manual guidance for {product_or_sku}: restart the device and "
        "check the quick-start guide."
    )


@mcp.tool()
def check_warranty_policy(product_or_sku: str) -> str:
    """Return warranty policy for a product or SKU."""
    return (
        f"{product_or_sku} includes a one-year limited warranty "
        "for manufacturing defects."
    )


if __name__ == "__main__":
    mcp.run()
