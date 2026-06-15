# 项目说明

## 注意

TTSA.py为spikingjelly模型，TTSA_old.py为snntorch模型


## 如何训练
python train_test_spi.py --seeds 3 --scenario highway-v0 --mode TTSA --warm-start --share-buffer --buffer-fraction 0.5


## 如何推理
python text_model.py --scenario highway-v0 --mode TTSA --seed 20 --checkpoint 90000 --num_episodes 5
