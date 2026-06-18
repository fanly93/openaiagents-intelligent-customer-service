"""Local product-support functions used by the stdio MCP server in Task 3.5."""


def lookup_product_manual(product_or_sku: str, question: str) -> str:
    return (
        f"Manual guidance for {product_or_sku}: restart the device and "
        "check the quick-start guide."
    )


def check_warranty_policy(product_or_sku: str) -> str:
    return (
        f"{product_or_sku} includes a one-year limited warranty "
        "for manufacturing defects."
    )
