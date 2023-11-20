# image-denoising

本文分为基本问题和选做问题两部分。基本问题中利用了Tikhonov正则化模型，将图像去噪问题描述为⼀个优化问题。并结合《矩阵论》课程上所学到的知识，用梯度下降法对问题进行求解，并分别选择了不同的常数步长以及Barzilai-Borwein步长、Adam步长作为梯度下降步长、选择不同的正则系数对结果的影响。在选做问题中，首先使用了与基本问题中同样的差分正则项，同样对基本问题中所选取的几个步长进行梯度下降优化。之后又分别尝试了$L_1$正则化、$L_2$正则化、弹性网以及核范数正则化，并将结果进行了对比，分析了不同的正则项对结果的影响。

## 运行方式
```shell
git clone https://github.com/SuperbRa1n/image-denoising.git
cd image_denoising
pip install -r requirements.txt
```

## 文件架构
* `results`目录用于存放输出的优化结果；
* `main`开头的文件分别为各种方式的主文件，使用不同的方法优化时只需要运行对应的文件即可;
* `evaluate.py`是包含如PSNR在内的各种评价指标的module；
* `image_proposing.py`是图片进行预处理的module；
* `gradient_descent.py`是梯度下降的module;
* `objective_function`是目标函数的module；
* `image.png`为进行加噪声并去噪的原始图片。