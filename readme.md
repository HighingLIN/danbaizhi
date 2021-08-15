# 蛋白质结构预测

## 运行方法
修改mode.py文件里面数据路径即可运行.

## 思路
把蛋白质的结构视为一句话，每个句子的精度为一个字母，然后embedding编码，再用宽视野的Conv1D提取每个字的局部特征，用MaxPooling1D再去提取局部特征（为了防止过拟合，所以pool_size比较大）.模型的灵感来自于天池蛋白质比赛的top3开源：https://github.com/yjh126yjh/TianChi_Protein-Secondary-Structure-Prediction    .

## 结语
知道这个比赛用的数据开源后有点下头，感谢DataWhale，希望日后有哪位大神看中我可以带我飞，啥苦活累活我都可以干.