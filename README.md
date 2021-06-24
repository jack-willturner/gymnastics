# gymNAStics
A "gym" style toolkit for building lightweight NAS systems. I know, the name is awful. 

## Supported operations

| Done | Tested | Op                  | Paper                                         | Notes                                                               |
| ---- | ------ | ------------------- | --------------------------------------------- | ------------------------------------------------------------------- |
| [x]  |        | conv                | -                                             | params: kernel size                                                 |
| [x]  |        | gconv               | AlexNet                                       | + params: group                                                     |
| [x]  |        | depthwise separable | [pdf](https://arxiv.org/pdf/1610.02357v3.pdf) |                                                                     |
| [x]  |        | mixconv             | [pdf](https://arxiv.org/pdf/1907.09595.pdf)   |                                                                     |
| [x]  |        | octaveconv          | [pdf](https://arxiv.org/pdf/1904.05049.pdf)   | Don't have a sensible way to include this as a single operation yet |
| [ ]  |        | shift               | [pdf](https://arxiv.org/pdf/1711.08141.pdf)   |                                                                     |
| [ ]  |        | ViT                 |                                               |                                                                     |
| [ ]  |        | Fused-MBConv        | [pdf](https://arxiv.org/pdf/2104.00298.pdf)   |                                                                     |
| [x]  |        | Lambda              | [pdf](https://arxiv.org/pdf/2102.08602.pdf)   |                                                                     |
| [ ]  |        |                     |                                               |                                                                     |
