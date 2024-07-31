# FlexGen-Lite: High-Throughput Generative Inference on Single GPU

## Introduction
FlexGen-Lite aims to replicate and extend the functionality of FlexGen, a system designed to enable high-throughput generative inference of large language models (LLMs) on a single GPU. By leveraging a sophisticated architecture that minimizes memory overhead and maximizes computational efficiency, FlexGen-Lite addresses the challenges of deploying LLMs on limited-resource hardware.

## Problem Statement
Deploying large language models on single GPU setups presents significant computational and memory challenges. Traditional methods struggle due to high memory demands and inefficiencies in data movement between GPU, CPU, and disk. These challenges result in performance bottlenecks, particularly for throughput-centric tasks.

## Proposed Solution
FlexGen-Lite introduces a hybrid CPU-GPU architecture that strategically divides computational tasks to optimize memory usage and processing efficiency. The CPU handles sequential tasks like the generation of Key (K), Query (Q), and Value (V) tensors, while the GPU is reserved for parallelizable tasks such as activation generation through Multi-Head Attention (MHA).

### Key Aspects of Enhanced Efficiency
1. **Efficient Tensor Management and Reduced I/O Costs**: Optimized layer-wise loading minimizes the need for frequent access to slower secondary storage, significantly reducing I/O costs.
2. **Strategic Offloading**: The zig-zag computational pattern aligns data more precisely with computational demands, enhancing overall system efficiency and throughput.
3. **Throughput Optimization via Batch and Block Sizing**: Dynamic batch sizing and block scheduling maximize GPU utilization, increasing throughput.

## Methodology
The implementation follows a hybrid CPU-GPU architecture:
- **CPU Responsibilities**: Initial data processing and attention calculations are performed on the CPU to avoid high I/O costs associated with transferring large KV caches between GPU and CPU.
- **GPU Responsibilities**: The GPU handles parallelizable tasks and activation handling, reducing the time required for data transfers.

### Optimized Data Flow
The zigzag traversal pattern for data management between CPU, GPU, and disk optimizes computational efficiency by minimizing unnecessary data movement.

## Results
Experiments were conducted using the OPT-1.3B language model on Google Colab. The results demonstrate significant improvements in throughput compared to traditional methods and the original FlexGen system. Detailed throughput analysis and comparisons are provided in the project report.

## Conclusion
FlexGen-Lite offers an optimized way to run highly batched high-throughput inference jobs on single GPU setups. This project enhances understanding of systems and large language models, especially for production model serving and inference pipelines.

## Installation
To install and run FlexGen-Lite, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/FlexGen-Lite.git
    cd FlexGen-Lite
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main script:
    ```bash
    python main.py
    ```

## Usage
Detailed instructions on how to use FlexGen-Lite can be found in the `docs` folder. Example usage:
```python
from flexgen_lite import FlexGenLite
model = FlexGenLite('path/to/model')
output = model.generate("Your input text here")
print(output)
```

## References
[FlexGen GitHub Repository](https://github.com/FMInference/FlexGen)  

[BitsAndBytes GitHub Repository](https://github.com/TimDettmers/bitsandbytes)

[Scalene GitHub Repository](https://github.com/plasma-umass/scalene)
