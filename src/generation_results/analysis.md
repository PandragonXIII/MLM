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
Average ASR:  0.411 
##### no defence
200 malicious text, 1 clean image, Average ASR:  0.335 (67/200);
200 malicious text, 4 adv image, Average ASR: 0.50875 (407/800)

## Qwen

##### no defence
200 malicious text, 1 clean image
Average ASR:  0.06 (12/200)


# Time complexity
## 100imgs+100texts
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/minigpt4_time_consumption.out)
with minigpt-4, processing phase takes 120.34s; generation: 3695.00s. total: 3815.34s

## direct generation 100 images+100 texts
[log](/home/xuyue/QXYtemp/MLM/src/generation_results/minigpt4_pure_generation.out)
without detection
total 3707.58s
107.76s faster.

