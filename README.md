# 基于深度学习的脑部疾病诊断
该程序提供了一种基于卷积神经网络诊断脑部医学影像是否具有某疾病的方法。在测试数据集上表现良好，参考诊断准确率达到0.9184.

请参考如下目录结构：
''' 
----HC                     #正常样本       
----patient                #异常样本
----checkpoints            #保存模型
----configs                #参数配置
--------config.py
----datasets               #数据处理
---------dataset.py        
--------preprocess.py      #数据预处理
--------splitdata.py       #划分测试集
----models
--------model.py
----logs                   #日志记录
----utils                  
--------dataloader.py
--------metrics.py         #评估函数
--------utils.py           
----train.py
----test.py
'''

运行顺序：
1. 运行preprocess.py
   将会将HC、patient里的数据进行预处理，储存在train/HC,train/patient内。
2. 运行splitdata.py
   将会从train/HC,train/patient中划分出测试集test/HC,test/patient。
3. 运行train.py
4. 运行test.py
