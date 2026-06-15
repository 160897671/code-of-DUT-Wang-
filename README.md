# code-of-DUT-Wang-
# TTSA - 基于脉冲神经网络的高速公路驾驶决策模型
- `TTSA.py`：基于 SpikingJelly 框架
- `TTSA_old.py`：基于 snnTorch 框架

## 训练

使用 `train_test_spi.py` 脚本进行训练。示例命令：

```bash
python train_test_spi.py \
    --seeds 3 \
    --scenario highway-v0 \
    --mode TTSA \
    --warm-start \
    --share-buffer \
    --buffer-fraction 0.5

## 推理

使用 `text_model.py` 脚本进行训练。示例命令：

```bash
python text_model.py \
    --scenario highway-v0 \
    --mode TTSA \
    --seed 20 \
    --checkpoint 90000 \
    --num_episodes 5

