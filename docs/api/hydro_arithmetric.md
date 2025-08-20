# hydro_arithmetric

The `hydro_arithmetric` module provides mathematical utilities for array operations in hydrological calculations.

## Functions

### random_choice_no_return

```python
def random_choice_no_return(arr: Union[list, np.ndarray], num_lst: list) -> list
```

Performs multiple sampling without replacement from an array.

**Args:**
- `arr`: The source array to sample from
- `num_lst`: List of integers specifying the number of elements to sample in each iteration

**Returns:**
- List of numpy arrays, where each array contains the sampled elements for that iteration

**Example:**
```python
import hydroutils as hu
import numpy as np

# Sample array
arr = [1, 2, 3, 4, 5]
num_lst = [2, 1]  # First sample 2 elements, then 1 element

# Perform sampling
result = hu.random_choice_no_return(arr, num_lst)
# result[0] contains 2 elements
# result[1] contains 1 element
```
