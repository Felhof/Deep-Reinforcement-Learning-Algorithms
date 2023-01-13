from typing import Tuple, Union


def get_dimension_format_string(dims: Union[int, Tuple[int, ...]], dtype: str = "float32") -> str:
    if isinstance(dims, int):
        return f"{dims}{dtype}"

    dim_list = list(dims)
    if len(dim_list) > 1 and dim_list[-1] == 1:
        dim_list.pop()

    if len(dim_list) == 1:
        return f"{dim_list[0]}{dtype}"

    return f"({','.join([str(d) for d in dim_list])}){dtype}"
