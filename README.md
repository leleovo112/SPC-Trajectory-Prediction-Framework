# SPC-Trajectory-Prediction-Framework

This repository contains the official implementation of the paper:

**“A neuro-symbolic framework for semantic-physical consistent trajectory prediction.”**

Published in *Expert Systems With Applications*, Volume 333 (2027), Article 133835.

- **Paper:** [10.1016/j.eswa.2026.133835](https://doi.org/10.1016/j.eswa.2026.133835)
- **Authors:** Yao Xiao, Lecong Huang, Chen Xiong
- **Corresponding author:** Chen Xiong
- **Affiliation:** School of Intelligent Systems Engineering, Sun Yat-sen University, Shenzhen, China

> **Note:** We are currently organizing and cleaning the source code to ensure that it is user-friendly, well documented, and fully reproducible. The complete replication package, including detailed installation, configuration, and evaluation instructions, will be released after the code has been carefully verified.

## Abstract

Trajectory prediction is pivotal for autonomous driving, yet deep-learning models can struggle in safety-critical, long-tail scenarios because their decisions are difficult to interpret and may violate physical or traffic constraints. This paper presents a hierarchical neuro-symbolic framework that combines rule-guided semantic reasoning with physically constrained trajectory generation.

An upper-level large language model (LLM) acts as a cognitive expert and uses structured Chain-of-Thought (CoT) reasoning to produce validated driving intentions. A lower-level data-driven execution module decodes these intentions into continuous trajectories subject to vehicle-dynamics, map-topology, and safety constraints. A semantic-physical verification and arbitration layer treats LLM outputs as semantic priors rather than unconditional commands, allowing physical feasibility to take precedence when conflicts occur.

## Key Features

- **LLM-Driven CoT Reasoning:** Uses structured prompts and rule-grounded reasoning to generate interpretable driving intentions.
- **Semantic-Physical Consistency:** Connects high-level semantic decisions with physically feasible trajectory generation.
- **Hierarchical Expert System:** Separates cognitive reasoning from low-level trajectory execution.
- **Safety-Priority Arbitration:** Resolves conflicts between semantic intentions and physical, map, or traffic constraints.
- **12-Class Meta-Action Interface:** Uses three lateral actions and four longitudinal actions to provide a compact semantic interface.
- **Interpretable Predictions:** Produces traceable reasoning paths and human-readable explanations.

## Key Results

The framework was evaluated on the **DeepAccident** safety-critical V2X benchmark.

- Accident-scenario collision rate: **1.22 ± 0.02%**
- Accident ADE: **1.62 ± 0.02 m**
- Accident FDE: **2.51 ± 0.04 m**
- Accident ADE improvement over MTR: **16.9%**
- Intention accuracy: **78.5 ± 0.45%**
- Obstacle avoidance success rate on the non-compliant-agent subset: **90.84%**

The reported results characterize the DeepAccident simulation setting and should not be interpreted as a universal performance guarantee across all real-world autonomous-driving domains.

## Framework Structure

The framework consists of four main stages:

1. **Multi-modal Feature Encoding**
   - Heterogeneous vectorization of agent states and HD-map elements.
   - Encoding of vehicle dynamics, lane topology, traffic-light states, and other scene attributes.

2. **Hierarchical Interaction Modeling**
   - Agent-agent interaction modeling.
   - Agent-map cross-attention.
   - Spatial, semantic, and safety-priority interaction features.

3. **LLM-Driven Cognitive Expert**
   - Rule-grounded semantic serialization.
   - Structured Chain-of-Thought reasoning.
   - Meta-action generation and parsing.
   - Logical and physical consistency verification.

4. **Constrained Trajectory Execution**
   - Intention-conditioned Transformer decoding.
   - Dynamic-feasibility constraints.
   - Map-compliance and safety-distance constraints.
   - Semantic-physical arbitration for conflicting intentions.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- DeepAccident dataset
- An LLM inference backend for the cognitive reasoning module

The exact installation and execution commands will be provided together with the complete replication package.

## Data Preparation

This framework is evaluated on the **DeepAccident** benchmark, a simulation-generated V2X dataset containing approximately 285k samples and diverse accident-prone interactive scenarios.

Due to dataset size and licensing conditions, please download the dataset from the official project page:

[DeepAccident Download Page](https://deepaccident.github.io/download.html)

After downloading, configure the dataset paths according to the configuration files that will be included in the released code package.

## Reproducibility Notes

The published experiments use the following settings:

- Random seed: **42**
- Observation horizon: **20 frames**
- Prediction horizon: **30 frames**
- Execution decoder: **6-layer Transformer**
- Hidden dimension: **256**
- Optimizer: **AdamW**
- Initial learning rate: **1 × 10⁻⁴**
- Weight decay: **0.01**
- GPT-4-Turbo decoding temperature: **0**

The LLM reasoning module is used through in-context inference. Cloud-LLM latency contributes to the reported end-to-end inference time and is not representative of millisecond-level onboard closed-loop control.

## Current Status

The source code is currently being prepared for public release. We are checking the implementation, configuration files, dataset processing scripts, pretrained model interfaces, and evaluation procedures to ensure that the released package is consistent with the published paper.

The complete release will include:

- Installation instructions;
- Dataset preparation scripts;
- Training and evaluation configurations;
- LLM prompt templates;
- Model checkpoints or checkpoint-loading instructions;
- Reproduction scripts for the main tables and figures;
- Detailed documentation for the semantic-physical verification module.

## Limitations

The current evaluation is limited to the DeepAccident simulation benchmark. Performance may differ on real-world datasets, dense scenes with many interacting agents, and rare compound maneuvers outside the predefined 12-action interface.

The current reasoning core relies on a cloud-hosted LLM, so exact bit-level reproducibility and millisecond-level onboard deployment are not guaranteed. The framework should currently be understood as an interpretable high-level semantic reasoning and safety-monitoring layer rather than a fully deployed autonomous-driving controller.

## Citation

```bibtex
@article{xiao2026neurosymbolic,
  title   = {A neuro-symbolic framework for semantic-physical consistent trajectory prediction},
  author  = {Xiao, Yao and Huang, Lecong and Xiong, Chen},
  journal = {Expert Systems With Applications},
  volume  = {333},
  pages   = {133835},
  year    = {2027},
  doi     = {10.1016/j.eswa.2026.133835}
}
