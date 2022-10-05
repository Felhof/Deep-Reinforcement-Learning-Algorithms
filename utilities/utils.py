def get_dimension_format_string(
    x_dim: int, y_dim: int = 1, dtype: str = "float32"
) -> str:
    if y_dim > 1:
        return f"({x_dim},{y_dim}){dtype}"
    if x_dim > 1 and y_dim == 1:
        return f"{x_dim}{dtype}"
    return dtype
