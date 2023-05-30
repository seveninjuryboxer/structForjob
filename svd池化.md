# svd池化

```python
import torch

class MyAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super(MyAdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        batch_size, channels, input_height, input_width = x.size()
        output_height, output_width = self.output_size
        # 乘以单位矩阵变换维度
        x = x.view(batch_size, channels, -1) * torch.eye(x.size(1)).type_as(x)
        # 计算平均池化
        x = torch.mean(x, dim=2, keepdim=True)
        # 反转维度顺序
        x = x.permute(0, 2, 1, 3)
        # 计算输出尺寸
        output_size = [output_height, output_width]
        # 进行双线性插值
        x = torch.nn.functional.interpolate(x, size=output_size, mode='bilinear', align_corners=True)
        # 再次反转维度顺序
        x = x.permute(0, 2, 1, 3)
        return x.squeeze()

# 实例化 MyAdaptiveAvgPool2d 类，并进行 forward 操作
adaptive_avg_pool = MyAdaptiveAvgPool2d([7, 7])
inputs = torch.randn(32, 64, 10, 9) # 输入数据 batch_size=32, channels=64, 高度=10, 宽度=9
outputs = adaptive_avg_pool(inputs)
print(outputs.size()) # 输出: torch.Size([32, 64, 7, 7])
```
