# NoC Simulator for Neural Network Execution

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概述

本项目提供了一个基于 Python 的仿真器，用于模拟神经网络在片上网络（NoC）架构上的执行过程。它允许用户探索不同的 NoC 配置、并行化策略（数据并行、模型并行、混合并行）以及容错机制对神经网络推理性能（主要关注延迟）的影响。

该仿真器旨在帮助研究人员和工程师：

*   理解 NoC 架构的基本原理及其对通信的影响。
*   评估不同并行计算策略在 NoC 环境下的优劣。
*   研究故障注入、错误检测和恢复策略对系统可靠性和性能的影响。
*   可视化 NoC 拓扑、链路状态和通信路径。

## 主要特性

*   **可配置的 NoC 架构:**
    *   目前支持二维 Mesh（`2D_mesh`）拓扑。
    *   可自定义 NoC 维度 (`dimensions`) 和 PE 数量 (`pe_count`)。
    *   可配置 PE 缓冲区大小 (`buffer_size`)。
*   **路由算法:**
    *   目前支持 XY 路由 (`routing_algo="XY"`)。
*   **神经网络建模:**
    *   简单的 `NeuralNetwork` 和 `Layer` 类来表示模型结构。
    *   支持定义层类型、神经元数量和权重（简化）。
*   **并行化策略:**
    *   **数据并行 (`data_parallel`):** 将输入数据分割到多个 PE 上，每个 PE 处理一部分数据。
    *   **模型并行 (`model_parallel`):** 将网络的不同层或部分分配给不同的 PE。
    *   **混合并行 (`hybrid`):** 结合数据并行和模型并行的策略。
    *   灵活的层到 PE 映射 (`create_mapping`)。
*   **容错机制:**
    *   可启用/禁用容错 (`fault_tolerance_enabled`)。
    *   支持链路故障注入 (`inject_fault`) 和移除 (`remove_fault`)。
    *   **错误检测方法 (`error_detection_method`):**
        *   错误检测码 (EDC) - 示例为奇偶校验。
        *   空间冗余 (Spatial Redundancy) - 示例为 TMR (Triple Modular Redundancy)。
        *   时间冗余 (Time Redundancy) - 示例为重复传输。
        *   *(心跳机制 `heartbeat` 在当前实现中用于链路检查，而非数据校验)*
    *   **错误恢复方法 (`error_recovery_method`):**
        *   重传 (`retransmission`)。
        *   重路由 (`rerouting`) - 尝试寻找替代路径。
        *   *(检查点 `checkpointing` 和 容错路由 `fault_tolerant_routing` 有占位符或集成在路由逻辑中)*。
*   **仿真与分析:**
    *   `run_simulation` 函数执行端到端的神经网络推理仿真。
    *   计算总延迟和每层延迟。
    *   详细的日志输出 (可选 `verbose=True`)。
*   **可视化:**
    *   使用 `networkx` 和 `matplotlib` 可视化 NoC 拓扑。
    *   高亮显示故障链路（红色）和当前活动通信链路（绿色）。

## 安装

1.  **克隆仓库:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **安装依赖:**
    建议使用虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install numpy matplotlib networkx
    ```
    或者，创建一个 `requirements.txt` 文件：
    ```txt
    numpy
    matplotlib
    networkx
    ```
    然后运行：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

直接运行 Python 脚本即可执行预定义的案例研究：

```bash
python your_script_name.py
```

脚本 (`your_script_name.py` 替换为你的文件名) 将依次执行以下案例研究：

1.  **数据并行:** 演示如何在所有 PE 上分割数据进行计算。
2.  **模型并行:** 演示如何将模型的不同层分配给特定 PE。
3.  **混合并行:** 演示结合数据和模型并行的策略。
4.  **容错:** 演示注入链路故障，并使用配置的错误检测/恢复机制进行处理，并与无故障情况进行比较。
5.  **不同 NoC 规模:** 分析 NoC 尺寸对通信延迟的影响。
6.  **不同错误处理方法:** 比较不同错误检测和恢复组合在故障场景下的性能。

### 自定义仿真

你可以修改脚本末尾的 `if __name__ == "__main__":` 部分，或者创建新的脚本来设置你自己的仿真场景：

```python
import numpy as np
# Import necessary classes (NoC, PE, Link, NeuralNetwork, Layer, run_simulation)

# 1. 配置 NoC
my_noc = NoC(topology="2D_mesh", dimensions=(3, 3), buffer_size=16,
             fault_tolerance_enabled=True, error_detection_method="EDC",
             error_recovery_method="retransmission")

# 2. 定义神经网络
layer1 = Layer(layer_type="dense", num_neurons=9)
layer2 = Layer(layer_type="dense", num_neurons=9)
my_nn = NeuralNetwork(layers=[layer1, layer2])

# 3. 准备输入数据
input_data = np.arange(9)

# 4. (可选) 注入故障
# my_noc.inject_fault((0, 0), (0, 1))

# 5. 选择映射策略和配置 (如果需要)
mapping_strategy = "data_parallel"
# model_parallel_config = [...] # For model or hybrid

# 6. 运行仿真
total_latency, final_output, per_layer_latencies = run_simulation(
    my_noc, my_nn, input_data,
    mapping_strategy=mapping_strategy,
    # model_parallel_config=model_parallel_config,
    verbose=True
)

# 7. 打印或分析结果
print("\nCustom Simulation Results:")
print(f"  Total Latency: {total_latency}")
print(f"  Final Output: {final_output}")
print(f"  Per-Layer Latencies: {per_layer_latencies}")

# 8. (可选) 清理故障
# my_noc.remove_fault((0, 0), (0, 1))
```

## 代码结构

*   `NoC`: 核心类，表示 NoC 架构，管理 PE、链路，并处理路由和数据包发送。
*   `PE`: 表示处理单元，包含输入/输出缓冲区和简化的处理逻辑。
*   `Link`: 表示 PE 之间的通信链路，具有延迟、带宽和故障状态。
*   `NeuralNetwork`: 表示神经网络模型，包含层列表和映射逻辑。
*   `Layer`: 表示神经网络中的一层。
*   `run_simulation`: 主仿真函数，协调 NoC、NN 和数据流。
*   `case_study_*` 函数: 演示不同场景的具体示例。

## 仿真结果摘要 (基于提供的输出)

*   **并行策略:**
    *   **数据并行:** 在 4x4 NoC 上，总延迟为 5 个周期。通信发生在第一层计算之后，数据从源 PE 广播到下一层的所有 PE。最终输出是多个 PE 处理结果的合并（当前实现简单地将所有输出连接起来）。
    *   **模型并行:** 在 4x4 NoC 上，将两层分别映射到 `(0,0)` 和 `(0,1)`，总延迟为 1 个周期（不包括计算时间，只计算通信）。通信仅发生在相邻层映射的 PE 之间。
    *   **混合并行:** 在 4x4 NoC 上，将两层分别映射到不同的 PE 组，总延迟为 3 个周期。通信发生在两组 PE 之间。
*   **容错:**
    *   在 4x4 NoC 上注入 `(0,0)` 到 `(0,1)` 的故障，并使用 `rerouting` 恢复策略。仿真显示，带故障的总延迟为 5 个周期。
    *   在移除故障后再次运行（无故障），总延迟为 4 个周期。
    *   **对比:** 故障注入和 `rerouting` 恢复导致了 1 个周期的额外延迟。*(注意：在提供的无故障运行结果中，出现了 "PE (0, 2) input buffer full" 的错误，这表明即使没有链路故障，数据并行下的广播通信也可能导致缓冲区溢出，需要调整缓冲区大小或流量控制策略)*。
*   **NoC 规模:**
    *   2x2 NoC: 总延迟 2 周期。
    *   4x4 NoC: 总延迟 5 周期。
    *   8x8 NoC: 总延迟 9 周期。随着 NoC 规模增大，对于相同的数据并行工作负载，平均通信距离增加，导致总延迟增加。*(注意：在 8x8 的结果中，再次出现了大量 "PE (4, 1) input buffer full" 错误，突显了大规模 NoC 中潜在的拥塞问题)*。
*   **错误处理方法:**
    *   EDC + 重传: 在 4x4 NoC 带故障时，总延迟 5 周期。
    *   空间冗余 + 重路由: 总延迟 5 周期。*(同样遇到了缓冲区溢出问题)*
    *   时间冗余 + 重传: 总延迟 5 周期。*(同样遇到了缓冲区溢出问题)*
    *   这些结果表明，对于当前的简单故障场景和恢复机制，总延迟相似，但实际效果可能受具体实现和流量模式影响。缓冲区溢出问题需要进一步研究。

## 可视化

仿真过程中会使用 `matplotlib` 生成 NoC 拓扑图。

*   节点代表 PE。
*   灰色边代表正常链路。
*   红色边代表注入的故障链路。
*   绿色边代表在当前层间通信步骤中活动的链路。

每次层处理和通信后，都会显示更新后的 NoC 状态图。

![Example Visualization Placeholder](placeholder_image_url.png)
*(建议在此处替换为实际的截图)*

## 未来工作与改进方向

*   **更多拓扑:** 实现 Torus、Fat Tree 等其他 NoC 拓扑。
*   **更复杂的路由算法:** 实现自适应路由、基于拥塞的路由等。
*   **详细的 PE 模型:** 加入更真实的计算模型（例如 MAC 操作计数）、内存访问模式。
*   **能量模型:** 估算通信和计算的能耗。
*   **流量控制:** 实现更复杂的流控机制（信用、门控等）以缓解拥塞。
*   **更真实的错误模型:** 模拟瞬态故障、永久性故障等。
*   **高级容错策略:** 实现更复杂的检查点/恢复、旁路路由等。
*   **集成真实模型:** 支持加载标准格式的神经网络模型（如 ONNX）。
*   **性能指标:** 添加吞吐量、带宽利用率等更多性能指标。

## 贡献

欢迎提交 Pull Requests 或创建 Issues 来报告错误、提出改进建议或贡献新功能。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
