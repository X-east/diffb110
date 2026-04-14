# test_output.py
epoch = 1
total_epochs = 2

# 补丁中的代码
print(f"\n{'='*60}")
print(f"Epoch {epoch+1}/{total_epochs}")
print(f"{'='*60}\n")

# 您的代码中的进度输出
_progress_epoch = (1, 2)
_epoch_str = f"Epoch {_progress_epoch[0]}/{_progress_epoch[1]}" 
print(f"\r[{_epoch_str}] 生成中: 4分子/卡, 50% (500/1000步)  ", end="", flush=True)