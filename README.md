![gymNAStics](figures/capybara.svg)

<p align="center">
  <!-- license -->
  <a href="https://tldrlegal.com/license/apache-license-2.0-%28apache-2.0%29">
      <img src="https://img.shields.io/github/license/jack-willturner/gymNAStics" alt="License" height="20">
  </a>
  <!-- CI status -->
  <a href="">
    <img src="https://img.shields.io/github/workflow/status/jack-willturner/gymNAStics/CI" alt="CI status" height="20">
  </a>
  <!-- Code analysis -->
  <img src="https://img.shields.io/lgtm/grade/python/github/jack-willturner/gymNAStics" alt="Code analysis" height="20">
  <!-- Getting started colab -->
  <a href="">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
  </a>
</p>

<p align="center">
  <i>A "gym" style toolkit for building lightweight NAS systems. I know, the name is awful. </i>
</p>


## Supported operations

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
