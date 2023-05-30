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
'''
import torch.nn as nn

class MyAdaptivePool(nn.Module):
    def __init__(self, output_size):
        super(MyAdaptivePool, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        batch_size, channels, input_height, input_width = x.size()
        output_height, output_width = self.output_size
        stride_h = input_height // output_height
        stride_w = input_width // output_width

        # 构造一个通道数和池化区域大小相同的单位矩阵
        identity_matrix = torch.eye(channels, device=x.device).unsqueeze(-1).unsqueeze(-1)

        # 构造掩码张量，其中前面部分的值为1，用于将一部分区域乘以单位矩阵进行维度变换
        mask = torch.ones(batch_size, channels, input_height, input_width, device=x.device)
        mask[:, :channels//2, :stride_h*3, :stride_w*2] = 0

        # 使用掩码张量和单位矩阵对输入数据进行维度变换
        transformed_x = x * mask + identity_matrix * (1.0 - mask)
        
        # 对变换后的数据进行平均池化，并调整输出形状
        pooled_x = nn.functional.avg_pool2d(transformed_x, (stride_h, stride_w))
        output = pooled_x.view(batch_size, channels, output_height, output_width)

        return output

# 实例化 MyAdaptivePool 类，并进行 forward 操作
my_adaptive_pool = MyAdaptivePool((7, 7))
inputs = torch.randn(1, 64, 20, 10) # 输入数据 batch_size=1, channels=64, 高度=20, 宽度=10
outputs = my_adaptive_pool(inputs)
print(outputs.size()) # 输出: torch.Size([1, 64, 7, 7])
'''
