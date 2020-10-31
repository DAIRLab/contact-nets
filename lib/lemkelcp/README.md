# lemkelcp

This is Python implementation of Lemke's algorithm for linear complementarity problems. Namely, it attempts to find a solution, `z` to the constraints:

```
z >= 0 
Mz + q >= 0 
z'(Mz + q) = 0
```

# Syntax and solution

```
sol = lemkelcp(M,q,maxIter)
```

Where `sol` is a tuple:

```
z,exit_code,exit_string = sol
```

|z                | exit_code | exit_string               |
|-----------------|-----------|---------------------------|
| solution to LCP |    0      | 'Solution Found'          |
| None            |    1      | 'Secondary ray found'     |
| None            |    2      | 'Max Iterations Exceeded' |    


# Examples

```
import numpy as np
import lemkelcp as lcp

M = np.array([[2,1],
              [0,2]])
q = np.array([-1,-2])

sol = lcp.lemkelcp(M,q)
```

Gives a solution

```
(array([ 0.,  1.]), 0, 'Solution Found')
```

```
M = np.array([[2,-1],
              [0,-2]])
q = np.array([-1,-2])

sol = lcp.lemkelcp(M,q)
```

gives a solution

```
(None, 1, 'Secondary ray found')
```


# Installation from PIP

```
pip install lemkelcp
```

# Installation from source



```
git clone https://github.com/AndyLamperski/lemkelcp.git
cd lemkelcp
python setupy.py install
```