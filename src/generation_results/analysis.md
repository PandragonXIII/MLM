test threshold: -0.0005046081542968749
## Llava
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/llava_eval_log.out)
llava spent 20.41s processing the inputs. But notice that it's because we reused same 5 images for all 201 text inputs. Therefore the practical time cost could be about 200 times larger.
And generating 1005 responses takes 2h 31min 31sec, 9091s in total, and average generation time is 9.05s
In addition, an extra 2 min is required for answer evaluation.
Average ASR:  0.273
Clean ASR: 0.34 (68/200)
Adv ASR: 0.25625 (205/800)
##### no defence
200 malicious text, 1 clean image
ASR: 0.38 (76/200)

## Blip
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/blip_eval_log.out)
similarly, processing the inputs takes 20.71s, which is not the case in practice.
*notice that blip sometimes has strange outputs like 1,10,100 or 10000...*
generating 1005 responses takes 1h 31min 28sec, 5488s in total, and average generation time is 5.46s
Clean ASR: 0.175 (35/200)
Adv ASR: 0.21875 (175/800)
Average ASR:  0.21
##### no defence
200 malicious text, 1 clean image
ASR: 0.115 (23/200)

## miniGPT-4
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/minigpt4_eval_log.out)
processing the inputs takes 24.43s, which is not the case in practice.
generating 1005 responses takes 10h 19min 49sec, 37189s in total, and average generation time is 37.00s
Clean ASR: 0.32 (64/200)
Adv ASR: 0.43375 (347/800)
Average ASR: 0.411 
##### no defence
200 malicious text, 1 clean image, Average ASR:  0.335 (67/200);
200 malicious text, 4 adv image, Average ASR: 0.50875 (407/800)

## Qwen
with defence
200 malicious text, 1 clean image, Average ASR: 0.04 
200 malicious text, 4 adv image, Average ASR: 0.055 
##### no defence
200 malicious text, 1 clean image, Average ASR: 0.06 (12/200)
200 malicious text, 4 adv image, Average ASR: 0.07125 (57/800)


# Time complexity
## 100imgs+100texts
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/minigpt4_time_consumption.out)
with minigpt-4, processing phase takes 120.34s; generation: 3695.00s. total: 3815.34s

## direct generation 100 images+100 texts
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/minigpt4_pure_generation.out)
without detection
total 3707.58s
107.76s faster.

## Qwen
100 malicious text, 100 clean image
without defence: 10min 2sec: 602s; 6.02s/query
with defence: 783.59s; 7.84s/query

# performance evaluation
#### llava-7b
with defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
llava_ans,7.3,5.3,4.5,8.6,6.8,0.0,5.9,0.0,[5.9]

no defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
llava_nodetect_ans,6.4,3.2,5.2,9.4,4.1,0.0,4.8,0.0,[4.8]

w/wo: 7.3/6.4&5.3/3.2&4.5/5.2&8.6/9.4&6.8/4.1&0.0/0.0&5.9/4.8&0.0/0.0&[5.9]/[4.8]

#### blip
with defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
blip_ans,10.5,9.9,5.4,13.9,10.1,3.5,8.9,0.0,[8.9]

no defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
blip_nodetect_ans,7.7,10.2,5.5,12.5,9.2,6.9,7.7,0.0,[7.7]

w/wo: 10.5/7.7&9.9/10.2&5.4/5.5&13.9/12.5&10.1/9.2&3.5/6.9&8.9/7.7&0.0/0.0&[8.9]/[7.7]

#### minigpt4
with defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
minigpt4_ans,15.0,7.1,6.7,9.6,15.1,1.9,11.2,0.0,[11.2]

no defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
minigpt4_nodetect_ans,15.3,9.2,11.7,16.9,13.1,3.8,12.4,0.0,[12.4]

w/wo: 15.0/15.3&7.1/9.2&6.7/11.7&9.6/16.9&15.1/13.1&1.9/3.8&11.2/12.4&0.0/0.0&[11.2]/[12.4]

#### qwen
with defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
qwen_ans,21.9,9.7,20.4,27.1,11.5,0.4,16.0,0.0,[16.0]

no defence:
    ,rec,ocr,know,gen,spat,math,total,std,runs
qwen_nodetect_ans,22.4,12.0,23.5,29.8,14.5,3.8,16.8,0.0,[16.8]

w/wo: 21.9/22.4&9.7/12.0&20.4/23.5&27.1/29.8&11.5/14.5&0.4/3.8&16.0/16.8&0.0/0.0&[16.0]/[16.8]
