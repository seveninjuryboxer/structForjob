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
[epoch 1] train_loss: 1.495  val_accuracy: 0.375
train epoch[2/50] loss:0.659: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[2/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 2] train_loss: 0.841  val_accuracy: 0.741
train epoch[3/50] loss:0.763: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[3/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 3] train_loss: 0.655  val_accuracy: 0.782
train epoch[4/50] loss:1.058: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[4/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 4] train_loss: 0.531  val_accuracy: 0.875
train epoch[5/50] loss:0.140: 100%|██████████| 63/63 [01:50<00:00,  1.75s/it]
valid epoch[5/50]: 100%|██████████| 27/27 [00:25<00:00,  1.06it/s]
[epoch 5] train_loss: 0.475  val_accuracy: 0.866
train epoch[6/50] loss:0.299: 100%|██████████| 63/63 [01:35<00:00,  1.52s/it]
valid epoch[6/50]: 100%|██████████| 27/27 [00:21<00:00,  1.25it/s]
[epoch 6] train_loss: 0.422  val_accuracy: 0.856
train epoch[7/50] loss:0.365: 100%|██████████| 63/63 [01:33<00:00,  1.49s/it]
valid epoch[7/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 7] train_loss: 0.431  val_accuracy: 0.880
train epoch[8/50] loss:0.126: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[8/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 8] train_loss: 0.410  val_accuracy: 0.889
train epoch[9/50] loss:0.383: 100%|██████████| 63/63 [01:33<00:00,  1.49s/it]
valid epoch[9/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 9] train_loss: 0.375  val_accuracy: 0.921
train epoch[10/50] loss:0.268: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[10/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 10] train_loss: 0.335  val_accuracy: 0.926
train epoch[11/50] loss:0.190: 100%|██████████| 63/63 [01:35<00:00,  1.51s/it]
valid epoch[11/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 11] train_loss: 0.318  val_accuracy: 0.880
train epoch[12/50] loss:0.072: 100%|██████████| 63/63 [01:35<00:00,  1.52s/it]
valid epoch[12/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 12] train_loss: 0.276  val_accuracy: 0.907
train epoch[13/50] loss:0.040: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[13/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 13] train_loss: 0.282  val_accuracy: 0.949
train epoch[14/50] loss:0.092: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[14/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 14] train_loss: 0.246  val_accuracy: 0.926
train epoch[15/50] loss:0.033: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[15/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 15] train_loss: 0.221  val_accuracy: 0.917
train epoch[16/50] loss:0.413: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[16/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 16] train_loss: 0.233  val_accuracy: 0.903
train epoch[17/50] loss:0.058: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[17/50]: 100%|██████████| 27/27 [00:21<00:00,  1.25it/s]
[epoch 17] train_loss: 0.256  val_accuracy: 0.931
train epoch[18/50] loss:0.587: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[18/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 18] train_loss: 0.200  val_accuracy: 0.935
train epoch[19/50] loss:0.038: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[19/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 19] train_loss: 0.306  val_accuracy: 0.935
train epoch[20/50] loss:0.485: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[20/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 20] train_loss: 0.183  val_accuracy: 0.940
train epoch[21/50] loss:0.144: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[21/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 21] train_loss: 0.206  val_accuracy: 0.954
train epoch[22/50] loss:0.085: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[22/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 22] train_loss: 0.229  val_accuracy: 0.958
train epoch[23/50] loss:0.131: 100%|██████████| 63/63 [01:39<00:00,  1.58s/it]
valid epoch[23/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 23] train_loss: 0.129  val_accuracy: 0.958
train epoch[24/50] loss:0.849: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[24/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 24] train_loss: 0.191  val_accuracy: 0.968
train epoch[25/50] loss:0.222: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[25/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 25] train_loss: 0.167  val_accuracy: 0.963
train epoch[26/50] loss:0.077: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[26/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 26] train_loss: 0.156  val_accuracy: 0.954
train epoch[27/50] loss:0.019: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[27/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 27] train_loss: 0.143  val_accuracy: 0.958
train epoch[28/50] loss:0.201: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[28/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 28] train_loss: 0.161  val_accuracy: 0.963
train epoch[29/50] loss:0.012: 100%|██████████| 63/63 [01:34<00:00,  1.49s/it]
valid epoch[29/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 29] train_loss: 0.186  val_accuracy: 0.963
train epoch[30/50] loss:0.120: 100%|██████████| 63/63 [01:35<00:00,  1.52s/it]
valid epoch[30/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 30] train_loss: 0.164  val_accuracy: 0.935
train epoch[31/50] loss:0.012: 100%|██████████| 63/63 [01:35<00:00,  1.51s/it]
valid epoch[31/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 31] train_loss: 0.188  val_accuracy: 0.968
train epoch[32/50] loss:0.012: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[32/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 32] train_loss: 0.185  val_accuracy: 0.968
train epoch[33/50] loss:0.072: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[33/50]: 100%|██████████| 27/27 [00:21<00:00,  1.25it/s]
[epoch 33] train_loss: 0.155  val_accuracy: 0.958
train epoch[34/50] loss:0.449: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[34/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 34] train_loss: 0.177  val_accuracy: 0.958
train epoch[35/50] loss:0.076: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[35/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 35] train_loss: 0.175  val_accuracy: 0.981
train epoch[36/50] loss:0.014: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[36/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 36] train_loss: 0.152  val_accuracy: 0.968
train epoch[37/50] loss:0.044: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[37/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 37] train_loss: 0.122  val_accuracy: 0.977
train epoch[38/50] loss:0.307: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[38/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 38] train_loss: 0.154  val_accuracy: 0.958
train epoch[39/50] loss:0.115: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[39/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 39] train_loss: 0.120  val_accuracy: 0.958
train epoch[40/50] loss:0.163: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[40/50]: 100%|██████████| 27/27 [00:21<00:00,  1.26it/s]
[epoch 40] train_loss: 0.170  val_accuracy: 0.972
train epoch[41/50] loss:0.019: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[41/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 41] train_loss: 0.111  val_accuracy: 0.968
train epoch[42/50] loss:0.028: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[42/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 42] train_loss: 0.109  val_accuracy: 0.944
train epoch[43/50] loss:0.019: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[43/50]: 100%|██████████| 27/27 [00:22<00:00,  1.23it/s]
[epoch 43] train_loss: 0.112  val_accuracy: 0.981
train epoch[44/50] loss:0.004: 100%|██████████| 63/63 [01:35<00:00,  1.51s/it]
valid epoch[44/50]: 100%|██████████| 27/27 [00:21<00:00,  1.25it/s]
[epoch 44] train_loss: 0.127  val_accuracy: 0.981
train epoch[45/50] loss:0.035: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[45/50]: 100%|██████████| 27/27 [00:21<00:00,  1.23it/s]
[epoch 45] train_loss: 0.142  val_accuracy: 0.981
train epoch[46/50] loss:0.017: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[46/50]: 100%|██████████| 27/27 [00:21<00:00,  1.25it/s]
[epoch 46] train_loss: 0.091  val_accuracy: 0.972
train epoch[47/50] loss:0.004: 100%|██████████| 63/63 [01:34<00:00,  1.51s/it]
valid epoch[47/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 47] train_loss: 0.074  val_accuracy: 0.972
train epoch[48/50] loss:0.001: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[48/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 48] train_loss: 0.129  val_accuracy: 0.981
train epoch[49/50] loss:0.059: 100%|██████████| 63/63 [01:35<00:00,  1.51s/it]
valid epoch[49/50]: 100%|██████████| 27/27 [00:21<00:00,  1.28it/s]
[epoch 49] train_loss: 0.097  val_accuracy: 0.981
train epoch[50/50] loss:0.048: 100%|██████████| 63/63 [01:34<00:00,  1.50s/it]
valid epoch[50/50]: 100%|██████████| 27/27 [00:21<00:00,  1.27it/s]
[epoch 50] train_loss: 0.087  val_accuracy: 0.981
Finished Training

Process finished with exit code 0
