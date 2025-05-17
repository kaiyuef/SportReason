# run_ircot_numsports.py
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline.active_pipeline import IRCOTPipeline   # ← 确认 import 路径
# 如果 repo 里是 flashrag.pipeline.ircot ，请按实际改

# ---------- 1. 读取配置 ----------
cfg = Config(
    config_file_path="FlashRAG/flashrag/config/numsports.yaml",
    # 仅把需要覆盖的键写进 dict；此处覆盖迭代轮数
    config_dict={"ircot": {"max_iter": 5}}
)

# ---------- 2. 载入数据 ----------
splits = get_dataset(cfg)
test_data = splits["test"]          # 你的数据集只有 test

# ---------- 3. 初始化 Pipeline ----------
ircot_pipe = IRCOTPipeline(
    config=cfg,
    max_iter=cfg["ircot"]["max_iter"]   # or直接写5
    # prompt_template=None  → 使用类中默认模板
)

# ---------- 4. 运行并评估 ----------
out_ds = ircot_pipe.run(test_data, do_eval=True)

print("Finished! 结果文件已写入:", cfg["save_dir"])
