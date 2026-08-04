export PYTHONPATH=/home/fengsicheng/Projects/SpecForge # 要让 specforge框架跑的是我们改的这版。如果之前pip安装的时候就是直接OK的话，这里这个可以注释掉


# 这个地方TODO替换一下你需要保存到的outputs路径
rm -rf /TODO/outputs/qwen3.5-4b-mmflash/control
rm -rf /TODO/outputs/qwen3.5-4b-mmflash/consumer-state

specforge train --config scripts/mmtraining_configs/qwen3.5-4b-mmflash_song.yaml