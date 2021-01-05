# fast-nnls

This is a Python implementation of the algorithm described in the paper
"A Fast Non-Negativity-Constrained Least Squares Algorithm" by
Rasmus Bro and Sumen De Jong.

Give a matrix `A` and a vector `y`, this algorithm solves `argmin_x ||Ax - y||`.

At the time of writing this, there are no readily available Python bindings
for this algorithm that I know of.
`scipy.optimize.nnls` implements the slower Lawson-Hanson version.

## Installation
From the top level `fast-nnls` directory, run

`python3 setup.py install`

## Usage

```python
from fastnnls import fnnls
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([1, 2, 3])
AtA = A.T.dot(A)
Aty = A.T.dot(y)
x = fnnls(AtA, Aty)
```

## Testing
Install `fastnnls` then run `python3 tests/test.py`.
