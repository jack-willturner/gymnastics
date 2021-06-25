## Models

This module provides an interface for constructing a `Model` from a `Genotype`. 

Models are composed of:
1. A backbone (currently just a ResNet)
2. A call (currently either a BasicBlock or a Bottleneck, filled with custom operations)

The custom operations are all implemented as `Layer`s in the `layers` folder. 