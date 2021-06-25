## Search spaces

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
- has the following ops: `conv3x3`, `conv1x1`, `identity`, `zeroize`, `avgpool3x3`

Here's how we can create the NAS-Bench-201 search space:

```python
class NASBench201Skeleton:
    def build_with_cell(self, cell: Cell) -> Skeleton:
        return ResNetCIFARSkeleton(cell, [3, 3, 3, 3])

skeleton = NASBench201Skeleton()

cell_space = CellSpace(
    ops=[Conv3x3, Conv1x1, AvgPool2d, Identity, Zeroize], num_nodes=4, num_edges=6
)

search_space = SearchSpace(skeleton, cell_space)
```


