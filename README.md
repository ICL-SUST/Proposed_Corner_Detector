# Proposed Corner Detector

这是论文所提 **SOGDD 相邻角点检测器**的模块化独立实现。

## 目录结构

```text
Proposed_Corner_Detector/
│
├── detector.py          # 主检测器
├── sogdd.py             # SOGDD 核与多方向滤波
├── response.py          # 局部自相关矩阵与响应计算
├── postprocess.py       # NMS、阈值与关键点筛选
├── utils.py             # 参数、图像读取与可视化工具
├── demo.py              # 单张图像测试
├── requirements.txt
└── README.md
```

## 核心角点响应

对局部区域内多方向 SOGDD 的绝对响应构造 Gram/自相关矩阵 \(M\)，
角点度量为：

```text
response = det(M) / (trace(M) + eps)
```

## 默认参数

```python
sigma = 1.2
num_directions = 4
neighbourhood_radius = 2
neighbourhood_shape = "disk"
nms_radius = 1.0
threshold_mode = "quantile"
threshold_value = 0.99
```

## 安装

```powershell
pip install -r .\requirements.txt
```

## 单张图片运行

把输入图片放到当前目录，例如 `Table.png`：

```powershell
python .\demo.py ".\Table.png" --output-dir ".\proposed_output" --save-response
```

输出：

```text
proposed_output/
├── keypoints.csv
├── keypoints_overlay.png
├── response.npy
├── valid_mask.npy
└── summary.json
```

## 在其他 Python 文件中调用

```python
from detector import DetectorConfig, proposed_detector

config = DetectorConfig(
    sigma=1.2,
    num_directions=4,
    neighbourhood_radius=2,
    nms_radius=1.0,
    threshold_mode="quantile",
    threshold_value=0.99,
)

result = proposed_detector("Table.png", config)

print(result.corner_num)
print(result.keypoints_xy)  # 坐标顺序为 (x, y)
print(result.scores)
```

## 使用论文中的绝对阈值

原稿实验若需要使用 \(T_h=1\times10^9\)，可以设置：

```python
config = DetectorConfig(
    threshold_mode="absolute",
    threshold_value=1e9,
)
```
