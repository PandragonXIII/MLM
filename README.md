![liiusitration](./mmexport1711295338457.png) 
![work folw](./mmexport1711332319841.png) 

在基于image+text 输入的模型上，希望找到一种防御越狱攻击的方法。
### Main
#### 方法1
假设有(harmful text+posioned img)和(harmful text+ clean img)的数据，分别计算img 和text 在embedding space 的向量。
**猜想**：harmful text posioned img 的embedding 向量之间的距离应该小于harmful text 和clean img 的embedding 向量之间的距离。
对于未知的输入，通过计算img 和text 的embedding 向量之间的距离（绝对值），可以判断是否是攻击样本。

#### 方法2
假设有(harmful text+noised img)的数据，将img通过降噪得到新图片img2。比较 img与text的Correlation，img2与text的Correlation。
**猜想**：攻击样本的text与img的embedding高度耦合。因此降噪后Corr应该降低
对于未知的输入，通过计算降噪前后corr降低的比例判断是否为攻击样本。

> diffusion-based denoiser可以用[这个](https://github.com/openai/improved-diffusion)





#### VISUAL ADVERSARIAL EXAMPLES JAILBREAK ALIGNED LARGE LANGUAGE MODELS
1. 文章提到训练出的对抗样本有通用性，是否意味着图片与文字在越狱的作用上不一定有强关联？
2. 文章中提到对图像的降噪会降低其ASR
3. 文中提到的对抗方法：
   1. 先对图像施加高斯噪声
   2. 利用Diffusion 重新生成原图像
   3. observe that all three noise levels effectively purify our visual adversarial examples
   4. the defense might falter when faced with more delicate adaptive attacks [ 27 ].
   5. 在评价方法上缺乏标准化
   6. Since it is known that adversarial examples are fundamentally
difficult to address and remain an unsolved problem after a decade of study, we ask the question: how can we achieve AI alignment without addressing adversarial examples in an adversarial environment? This challenge is concerning, especially in light of the emerging trend toward multimodality in frontier foundation models


#### A Mutation-Based Method for Multi-Modal Jailbreaking Attack Detection
1. key observation is that attack queries inher-ently possess less robustness compared to benign queries.



### Step1
2024.3.25
check the cosine similarity of embeddings between adversarial images and malicious prompts/adversarial images and benign prompts.( Intuition check)
> Questions:
> Harmbench 的text+VISUAL ADVERSARIAL EXAMPLES? **yes, plus benign text**
> 是生成的新图片还是现有的图片？**existing**
> 

TODO:
* [x] adversarial images and clean images
* [x] encoder of LLaVa
  * [x] text
  * [x] img
* [x] algorithm to calculate cosine similarity
* [x] automamtic runnig
* [ ] data analysis
  * [x] 16扰动
  * [ ] other


### UPD 3.29
由于目前得到的embedding后的图片和文本的维度不同([576,4096] v.s. [50,4096]), 希望得到统一维度后再进行cosine similarity的计算。
以下是两种方法：
1. 通过decoder将图片解码后再通过[UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)得到向量。文本直接通过UAE得到向量。
2. 将emcode后的向量做平均得到句向量([1,4096])，再计算cosine similarity
3. 看看能不能用CLIP直接得到一维向量

### UPD 3.30
直接用均值方法得到了[1,4096]维的*句向量*，与最低扰动的图像之间测试余弦相似度，得到结果有一定的显著性
明天继续进行更多测试+写周报


```
发生异常: RuntimeError
Parent directory /home/xuyue/QXYtemp/src/embedding does not exist.
  File "/home/xuyue/QXYtemp/get_embed.py", line 72, in <module>
    torch.save(img_embed_list, f"{source_dir}/embedding/{img_save_filename}.pt")
RuntimeError: Parent directory /home/xuyue/QXYtemp/src/embedding does not exist.
```