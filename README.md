<div align="center">

# Cost-of-Pass: An Economic Framework for Evaluating Language Models
[![arXiv](https://img.shields.io/badge/arXiv-2504.13359-b31b1b.svg?style=flat&logo=arxiv)](https://arxiv.org/abs/2504.13359)
[![Benchmark](https://img.shields.io/badge/Benchmark-HuggingFace-ffcc00.svg?style=flat&logo=huggingface)](https://huggingface.co/CostOfPass)

🚧 **This repository and benchmark is under construction. Code and details coming soon!** 🚧
</div>


## Index
- [Overview](#overview)
- [Citation](#citation)

## Overview

<div align="center">
    <img src="framework_overview.png" alt="Framework Overview" style="width: 100%;"/>
</div> <br>
The widespread adoption of AI systems in the economy hinges on their ability to generate economic value that outweighs their inference costs. Evaluating this tradeoff requires metrics that account for both performance and costs.

We propose a framework grounded in production theory for evaluating language models by combining accuracy and inference cost. We introduce **Cost-of-Pass**, the expected monetary cost of generating a correct solution. We then define the **Frontier Cost-of-Pass** as the minimum Cost-of-Pass achievable across available models or *the human expert*, using the approximate cost of hiring an expert.

With our framework, we quantify the economic benefit that language models provide over an human expert baseline. We then track the evolution of cost-efficiency over the past year across different task types, evaluate the essentialness of various model innovations, and assess the economic value of common inference-time techniques.

Our findings point to clear trends in cost-efficiency across model classes and task types, reflecting the broader dynamics of innovation in the field. These patterns, and the shifts we've observed over time, offer a window into how economic value is increasingly shaped by model-level advances rather than surface-level improvements.

## Citation
If you find our work useful, please consider citing:
```bibtex
@misc{erol2025costofpass,
      title={Cost-of-Pass: An Economic Framework for Evaluating Language Models}, 
      author={Mehmet Hamza Erol and Batu El and Mirac Suzgun and Mert Yuksekgonul and James Zou},
      year={2025},
      eprint={2504.13359},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2504.13359}, 
}
```
