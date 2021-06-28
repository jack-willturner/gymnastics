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



## A walkthrough 

When we call `searchspace.sample_random_architecture` it does two things: it first generates a random cell from the cell space (via `CellSpace.generate_random_cell`), and then it initialises a skeleton using the cell. 

`CellSpace.generate_random_cell` constructs a `Cell`, which is really just a tuple of `Node`s and `Edge`s. The `Edge` is the important bit, and it carries an `Op` with it. The `Op` will need to be configured by the skeleton (e.g. if it's a ResNet style skeleton with four stages of increasing channels/decreasing spatial dimensions, the `Op`s will need to be configured to have the right number of input/output channels and strides.)

