# 基于深度学习的脑部疾病诊断
该程序提供了一种基于卷积神经网络诊断脑部医学影像是否具有某疾病的方法。在测试数据集上表现良好，参考诊断准确率达到0.9184.

请参考如下目录结构：
  ----HC
  ----patient
  ----checkpoints
  ----configs
  --------config.py
  ----datasets
  ---------dataset.py
  --------preprocess.py
  --------splitdata.py
  ----models
  --------model.py
  ----logs
  ----utils
  --------dataloader.py
  --------metrics.py
  --------utils.py
  ----train.py
  ----test.py

运行顺序：
1. 运行preprocess.py
   将会将HC、patient里的数据进行预处理，储存在train/HC,train/patient内。
2. 运行splitdata.py
   将会从train/HC,train/patient中划分出测试集test/HC,test/patient。
3. 运行train.py
4. 运行test.py
