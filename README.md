![logo](https://raw.githubusercontent.com/rainorangelemon/pygma_sphinx_theme/master/pygma_sphinx_theme/static/img/text_logo.png)

-----------------------

**[Documentation](https://pytorch-geometric-multiagent.readthedocs.io/en/latest/)**

The official repo for the CoRL 2022 paper 'Learning Control Admissibility Models with Graph Neural Networks for Multi-Agent Navigation' [[project page](https://rainorangelemon.github.io/CoRL2022/)]

<!--The current repo only includes GNN for control. For planning methods such as CBS and SIPP, please stay tuned.-->

The ultimate goal is to provide a benchmark and a handy tool for GNN researchers to conduct evaluations properly and fairly for multi-agent tasks.

**Note: The current repo is actively under maintenance.**

## Installation

```bash
conda create -n pygma python=3.8
conda activate pygma
# install pytorch, modify the following line according to your environment
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# install torch geometric, refer to https://github.com/pyg-team/pytorch_geometric
conda install pyg -c pyg
# install pyg_multiagent
pip install pyg_multiagent
```
