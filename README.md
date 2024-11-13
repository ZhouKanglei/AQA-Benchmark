<div align="center">
  <h1> A Comprehensive Benchmark for Action Quality Assessment </h1>
</div>
  
<div>&nbsp;</div>


Welcome to the AQA Benchmark repository. This repository contains the necessary code and resources to run and evaluate the AQA benchmark.

## 👀 Model Zoo

| Algorithm | Publisher | Paper | Supported Datasets |
|-----------|-------------|------|-------------------|
| [USDL/MUSDL](https://github.com/nzl-thu/MUSDL) | CVPR'20 | [arXiv](https://arxiv.org/abs/2006.07665) | [MTL-AQA](https://github.com/yuxumin/CoRe) |
| [CoRe](https://github.com/yuxumin/CoRe) | ICCV'21 | [arXiv](https://arxiv.org/pdf/2108.07797) | [MTL-AQA](https://github.com/yuxumin/CoRe) |
| [GDLT](https://github.com/xuangch/CVPR22_GDLT) | CVPR'22 | [pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_Likert_Scoring_With_Grade_Decoupling_for_Long-Term_Action_Assessment_CVPR_2022_paper.pdf) | [MTL-AQA](https://github.com/yuxumin/CoRe) |
| [HGCN](https://github.com/yuxumin/CoRe) | TCSVT'23 | [pdf](https://zhoukanglei.github.io/publication/hgcn_aqa/HGCN_AQA.pdf) | [MTL-AQA](https://github.com/yuxumin/CoRe) |
| [CoFInAl](https://github.com/ZhouKanglei/CoFInAl_AQA) | IJCAI'24 | [arXiv](https://arxiv.org/abs/2404.13999) | [MTL-AQA](https://github.com/yuxumin/CoRe) |

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
    Place your datasets in a proper directory. You may need to change the data path for the configs provided in the `./configs` directory.

4. **Download I3D Pre-Trained Weight**:
    You can download the I3D pre-trained weights from the following link: [I3D Pre-Trained Weight](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth). After downloading, place the weight in the `./weights` directory.

5. **Train Models**:
    Use the training scripts in the `./scripts` directory to train your models. For example:
    ```bash
    python main.py \
        --config configs/{your config}.yaml \
        --exp_name {optional} \ 
        --gpus 1  \
        --phase {train/test, default is train}
    ```

6. **Evaluate Models**:
    After training, evaluate your models using the evaluation scripts:
    ```bash
    python main.py \
        --config configs/{your config}.yaml \
        --exp_name {optional} \ 
        --gpus 1  \
        --phase {train/test, default is train}
    ```

7. **View Results**:
    The results of your evaluations will be stored in the `./outputs/config_name/exp_name` directory. You can analyze and visualize these results as needed.

## 🤝 Contributing


We welcome contributions! Please fork the repository and submit a pull request with your changes.


## 📞 Contact Us

For any questions or issues, please open an issue on GitHub or reach out to us via email at [zhoukanglei{at}qq.com](mailto:zhoukanglei@qq.com).

Happy benchmarking!