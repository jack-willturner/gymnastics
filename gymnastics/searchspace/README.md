## Search spaces

A `SearchSpace` is a tuple of:
1. A `CellSpace`
2. A `Skeleton`

It can also optionally be connected to a tabular database (like NAS-Bench-101).

We also provide a few example `CellSpace`s and `Skeleton`s. 

For instance, the NAS-Bench-201 `CellSpace`: 
- has 1 input node 
- has 1 output node 
- has 2 intermediate nodes
- is fully connected
- has the following ops: conv3x3, conv1x1, identity, zeroize, avg pool 

