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

## Input/output channel solver

The configuration of the number of input/output channels is a little complex. In principle every cell should:

   1. Have a number of input channels 
   2. Have a number of output channels (to be multiplied by an expansion factor)
   3. Any edges that are connected to the input should have `input_channels` number of input channels
   4. Any edges that are connected to the output should have `output_channels` number of output channels 
   
There are a few edge cases which we have to have special rules for:

**Edge case 1**: The first operation is an "identity" operation which doesn't properly expand the number of channels. Solution: force the proceeding cells to take the reduced number of channels as input.

**Edge case 2**: The last operation is an "identity" operation which doesn't properly expand the number of channels. Solution: force the preceding cell to have the correct number of output channels. 

**Edge case 3**: A node receives multiple inputs from edges which produce different number of channels. Solution: topologically order the edges and set all edges to share the number output channels with the first edge.

**Edge case 4**: Pooling where `in_channels` != `out_channels`. Solution: include a projection layer (normally `conv1x1`).

**Edge case 5**: A layer with a stride > 1 and an `identity` are connected to the same output node. Solution: NAS-Bench-201/DARTS use a Skip connection with a `FactorizedReduce` layer.