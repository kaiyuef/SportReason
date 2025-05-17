# run_r1searcher_numsports.py
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import ReasoningPipeline   # ← key change

# ---------- 1. Load config ----------
cfg = Config(
    config_file_path="FlashRAG/flashrag/config/R1-searcher.yaml",
)

# ---------- 2. Load data ----------
splits = get_dataset(cfg)
test_data = splits["test"]

# ---------- 3. Initialize pipeline ----------
pipe = ReasoningPipeline(cfg)

# ---------- 4. Run & evaluate ----------
out_ds = pipe.run(test_data, do_eval=True)
print("Finished! 结果文件已写入:", cfg["save_dir"])
