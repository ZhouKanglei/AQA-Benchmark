<div align="center">
  <h1> A Comprehensive Benchmark for Action Quality Assessment </h1>
</div>
  
<div>&nbsp;</div>

[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://zhoukanglei.github.io/AQA-Survey)
[![Page Views Count](https://badges.toozhao.com/badges/01JDKP6H9FYE9WEX7ADV27SFW4/green.svg)](https://badges.toozhao.com/stats/01JDKP6H9FYE9WEX7ADV27SFW4 "Get your own page views count badge on badges.toozhao.com")


Welcome to the AQA Benchmark repository. This repository contains the necessary code and resources to run and evaluate the AQA benchmark.

## 👀 Model Zoo

| Algorithm | Publisher | Paper | Supported Datasets | Note |
|-----------|-------------|------|-------------------|-------------|
| [(M)USDL](https://github.com/nzl-thu/MUSDL) | CVPR'20 | [arXiv](https://arxiv.org/abs/2006.07665) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Uncertainty-Aware Direct Regression |
| [CoRe](https://github.com/yuxumin/CoRe) | ICCV'21 | [arXiv](https://arxiv.org/pdf/2108.07797) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Contrastive Regression |
| [GDLT](https://github.com/xuangch/CVPR22_GDLT) | CVPR'22 | [pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Likert_Scoring_With_Grade_Decoupling_for_Long-Term_Action_Assessment_CVPR_2022_paper.pdf) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Transformer-Based Direct Regression |
| [HGCN](https://github.com/yuxumin/CoRe) | TCSVT'23 | [pdf](https://zhoukanglei.github.io/publication/hgcn_aqa/HGCN_AQA.pdf) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | GCN-Based Direct Regression |
| [DAE](https://github.com/Luciferbobo/DAE-AQA) | NCAA'24 | [arXiv](https://arxiv.org/abs/2111.11029) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Uncertainty-Aware Direct Regression |
| [T2CR](https://github.com/XuHuangbiao/T2CR) | INFS'24 | [DOI](https://www.sciencedirect.com/science/article/abs/pii/S0020025524002603) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Contrastive Regression |
| [CoFInAl](https://github.com/ZhouKanglei/CoFInAl_AQA) | IJCAI'24 | [arXiv](https://arxiv.org/abs/2404.13999) | [MTL-AQA](https://github.com/ParitoshParmar/MTL-AQA), [FineDiving](https://github.com/xujinglin/FineDiving), [AQA-7](http://rtis.oit.unlv.edu/datasets/), [RG](https://github.com/qinghuannn/ACTION-NET), [Fis-V](https://github.com/chmxu/MS_LSTM), [LOGO](https://github.com/dx199771/Interpretability-AQA?tab=readme-ov-file#long-form-group-logo) | Direct Regression |

## 📂 Repository Structure

The repository is organized as follows:

```
AQA-benchmark/
├── data/                   # Contains the datasets used for benchmarking
├── scripts/                # Includes scripts for training and evaluation examples
├── models/                 # Directory for storing model architectures and checkpoints
├── outputs/                # Stores the results of the benchmark evaluations
├── datasets/               # Contains all the datasets used in the project
├── criterion/              # Includes the criteria or loss functions used in the project
├── utils/                  # Holds utility scripts and helper functions
├── data-folder-tree.txt    # Organize the data directory as the same as the file
├── README.md               # This file, providing an overview of the repository
```

## 📘 How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ZhouKanglei/AQA-Benchmark
    cd AQA-benchmark
    ```

2. **Install Dependencies**:
    Ensure you have Python and the necessary libraries installed. You can install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**:
    - Create the necessary directories and subdirectories as specified in `data-folder-tree.txt`.
    - Download the datasets using the recommended links provided in the **Model Zoo** section. Move your datasets into the appropriate directories.
    - Ensure that the data paths in your configuration files in the `./configs` directory match the new directory structure.

4. **Download I3D Pre-Trained Weight**:
    You can download the I3D pre-trained weight from the following link: [I3D Pre-Trained Weight](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth). After downloading, place the weight in the `./weights` directory.

5. **Train Models**:
    Use the training scripts in the `./scripts` directory to train your models. For example:
    ```bash
    python main.py \
        --config configs/{your config}.yaml \
        --exp_name {optional} \ 
        --gpus 1  \
        --phase {train/test, default is train} \
        --class_idx {optional, for rg, fis-v, seven}
    ```

6. **Evaluate Models**:
    After training, evaluate your models using the evaluation scripts:
    ```bash
    python main.py \
        --config configs/{your config}.yaml \
        --exp_name {optional} \ 
        --gpus 1  \
        --phase {train/test, default is train} \
        --class_idx {optional, for rg, fis-v, seven}
    ```

7. **View Results**:
    The results of your evaluations will be stored in the `./outputs/config_name/exp_name` directory. You can analyze and visualize these results as needed.

## 🤝 Contributing


We welcome contributions! Please fork the repository and submit a pull request with your changes.


## 📞 Contact Us

For any questions or issues, please open an issue on GitHub or reach out to us via email at [z h o u k a n g l e i {at} q q . c o m](mailto:zkl@qq.com) (Please remove the spaces in the email address before sending your message).

Happy benchmarking!

