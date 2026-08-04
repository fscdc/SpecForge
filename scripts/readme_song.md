昨天的ocr_vqa可以再跑一遍download，如果报错随时叫我。

然后把我传给你的regen后的数据放在/TODO/regen_data/下。其中传给你的这版本已经把图片根路径改为和你发我的一致：

运行下面即可进行训练，其中需要把`mmflash_training_song.sh` 和`mmtraining_configs/qwen3.5-4b-mmflash_song.yaml`中的路径改为实际需要存储的路径（只需把TODO部分统一替换即可）。当前是按照8卡的配置来写的

```
bash scripts/mmflash_training_song.sh
```


下面是可能在跑的时候遇到的问题：可能需要执行下面的脚本，然后可能需要一个pip install的报错，但是这个我记不住名字，但是解决办法会在报错时候显示，包括下面这个指令也会在报错过程中显示。具体还要看cuda13环境的报错情况。但是应该报错只有2-3次
```
bash scripts/apply_sglang_spec_capture_patch.sh
```
