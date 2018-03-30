# task2-pytorch_neural_dialogue_generation

标签（空格分隔）： python pytorch

---

## 一、项目说明
   利用pytorch实现论文 
   
   Li J, Monroe  W, Shi T, et al. Adversarial  learning  for neural  dialogue  generation[J].  arXiv  preprint arXiv:1701.06547, 2017.
   
   使用数据集：t_given_s_dialogue_length2_6 （http://nlp.stanford.edu/data/OpenSubData.tar）
   
## 二、代码文件说明
1. gen文件夹：实现生成器的模型及训练文件
   
   gen/seq2seq.py: 实现生成器的模型文件，用的是seq2seq模型，包括编码器、解码器、attention机制、teacher forcing.

   gen/generator.py: 实现有关生成器的训练部分程序，包括
   
   a. 生成器数据预处理(prepare_data_new，get_batch)
   
   b. 模型预训练(train)
   
   c. 模型解码测试(test_decoder)
   
   d. 利用判别器的reward训练生成器(train_with_reward)
   
   e. 利用teacher-forcing训练生成器(teacher_forcing)
   
   f. 利用生成器生成送入到判别器的训练数据(gen_disc_data)
   
2.  gen_data文件夹：放置生成器的训练数据 t_given_s_dialogue_length2_6.txt 和词汇表 movie_25000
3.  disc文件夹: 实现判别器的模型和训练过程

    disc/hier_rnn_model.py: 实现判别器模型，使用层次rnn模型，第一层分别对answer和query执行lstm,第二层对第一层两者提取出的信息再使用一次lstm,再经过线性映射进行二分类。

    disc/hier_disc.py: 实现判别器的训练，包括
    
    a. 数据预处理(prepare_data, get_batch)
    
    b. 判别器预训练(hier_train)
    
    c. 判别器的单步训练(disc_step)
    
4.  disc_data文件夹： 存放利用生成器和原始数据的生成的判别器的训练数据train.query, train.answer, train.gen文件
5.  utils文件夹
    
    utils/conf.py: 生成器和判别器的超参数配置文件
    
    utils/data_utils: 存放数据预处理的通用函数和变量

6.  main.py文件: 实现生成器和判别器联合训练的整个过程。

## 三、 训练流程
  分别执行main.py中的如下函数
    
1.  gen_pre_train: 预训练生成器模型
   
2.  gen_test: 生成器测试

3.  gen_disc(): 生成判别器的训练数据
    
4.  disc_pre_train: 预训练判别器模型

5.  al_train: 利用生成器和判别器预训练模型进行联合训练，实现论文中的算法
    ![image.png-53.8kB][1]


  [1]: http://static.zybuluo.com/njuzrs/0o9wsa024vzs8txe6oejtfs5/image.png