import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# data_provider/data_loader.py

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler
#
#
# class PHM_MergedMultivariateNpy(Dataset):
#     """
#     PHM walk-level dataset for Timer-XL (multivariate):
#       - Each run file is (T=315, C=8): [7 aggregated channels..., wear]
#       - Train/Val: use c1 + c4
#       - Test: use c6
#       - wear MUST be the last column (so --covariate will supervise only wear)
#
#     Windowing follows Timer-XL next-token objective:
#       seq_x: [seq_len, C]
#       seq_y: shifted target aligned to next-token prediction
#     """
#
#     def __init__(self, root_path, data_path=None, flag='train',
#                  size=None, scale=True, nonautoregressive=False,
#                  subset_rand_ratio=1.0, split_ratio=(0.7, 0.1, 0.2),**kwargs):
#         """
#         size: [seq_len, input_token_len, output_token_len, test_pred_len(optional)]
#               通常你会用 seq_len=96, input_token_len=24, output_token_len=24
#         split_ratio: (train, val, test) 仅用于 c1/c4 内部划分 train/val
#                      这里 test 不用（因为我们用 c6 作为跨刀具测试）
#         """
#         assert size is not None and len(size) >= 3
#         self.seq_len = int(size[0])
#         self.input_token_len = int(size[1])
#         self.output_token_len = int(size[2])
#
#         self.scale = scale
#         self.nonautoregressive = nonautoregressive
#         self.subset_rand_ratio = subset_rand_ratio
#
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         assert flag in type_map
#         self.set_type = type_map[flag]
#
#         # 固定：c1+c4 训练，c6 测试
#         self.train_runs = ['c1.npy', 'c4.npy']
#         self.test_runs = ['c6.npy']
#
#         # 允许你未来扩展：如果 data_path 传了列表，就用它覆盖默认
#         # e.g. data_path="c1.npy,c4.npy,c6.npy"
#         if data_path is not None and isinstance(data_path, str) and data_path.endswith('.npy') and ',' in data_path:
#             parts = [p.strip() for p in data_path.split(',') if p.strip()]
#             # 约定：前两个是 train runs，最后一个是 test run
#             if len(parts) >= 3:
#                 self.train_runs = parts[:-1]
#                 self.test_runs = [parts[-1]]
#
#         # 1) load all needed runs
#         self.raw = {}
#         for fn in (self.train_runs + self.test_runs):
#             fp = os.path.join(root_path, fn)
#             arr = np.load(fp)
#             if arr.ndim != 2:
#                 raise ValueError(f'{fp} should be 2D, got {arr.shape}')
#             self.raw[fn] = arr.astype(np.float32)
#
#         # 2) fit scaler only on TRAIN parts of c1+c4 (avoid leakage)
#         self.scaler = StandardScaler()
#         if self.scale:
#             train_slices = []
#             for fn in self.train_runs:
#                 arr = self.raw[fn]
#                 L = len(arr)
#                 n_train = int(L * split_ratio[0])
#                 # 用 train 段拟合 scaler
#                 train_slices.append(arr[:n_train])
#             fit_mat = np.concatenate(train_slices, axis=0)
#             self.scaler.fit(fit_mat)
#
#             # transform all runs
#             for fn in self.raw:
#                 self.raw[fn] = self.scaler.transform(self.raw[fn])
#
#         # 3) build per-flag segments (avoid cross-run windows)
#         if self.set_type in (0, 1):  # train / val from c1+c4
#             use_runs = self.train_runs
#         else:  # test from c6
#             use_runs = self.test_runs
#
#         self.data_list = []
#         self.n_window_list = []
#
#         for fn in use_runs:
#             arr = self.raw[fn]
#             L = len(arr)
#
#             if self.set_type == 2:
#                 border1, border2 = 0, L
#             else:
#                 # 只在 c1/c4 内部分 train/val，参考常用做法：val 起点往前留 seq_len 防止窗口不够
#                 n_train = int(L * split_ratio[0])
#                 n_val = int(L * split_ratio[1])
#                 # n_test = L - n_train - n_val  # 这里不用
#
#                 if self.set_type == 0:   # train
#                     border1, border2 = 0, n_train
#                 else:                    # val
#                     border1 = max(0, n_train - self.seq_len)
#                     border2 = min(L, n_train + n_val)
#
#             seg = arr[border1:border2]
#
#             # 走刀序列太短则跳过
#             if len(seg) < self.seq_len + self.output_token_len:
#                 continue
#
#             self.data_list.append(seg)
#
#             # 每个 seg 里能取多少个滑窗（按时间）
#             n_timepoint = len(seg) - self.seq_len - self.output_token_len + 1
#             cum = n_timepoint if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_timepoint
#             self.n_window_list.append(cum)
#
#         if len(self.n_window_list) == 0:
#             raise RuntimeError("No windows created. Check seq_len/output_token_len vs data length.")
#
#         # subset for quick debug (optional)
#         if self.set_type == 0 and self.subset_rand_ratio < 1.0:
#             self._subset_len = max(1, int(len(self) * self.subset_rand_ratio))
#         else:
#             self._subset_len = None
#
#     def __len__(self):
#         total = self.n_window_list[-1]
#         return self._subset_len if self._subset_len is not None else total
#
#     def __getitem__(self, index):
#         assert index >= 0
#
#         # quick subset mode: randomly sample
#         if self._subset_len is not None:
#             index = np.random.randint(0, self.n_window_list[-1])
#
#         # locate segment
#         dataset_index = 0
#         while index >= self.n_window_list[dataset_index]:
#             dataset_index += 1
#
#         prev_cum = self.n_window_list[dataset_index - 1] if dataset_index > 0 else 0
#         local_i = index - prev_cum
#
#         data = self.data_list[dataset_index]
#
#         s_begin = local_i
#         s_end = s_begin + self.seq_len
#         r_begin = s_begin + self.input_token_len
#         r_end = s_end + self.output_token_len
#
#         seq_x = data[s_begin:s_end, :]          # [seq_len, C=8]
#         seq_y = data[r_begin:r_end, :]          # [seq_len + out - in, C]
#
#         # y_mark / x_mark: Timer-XL 这里用不到，给全 0 即可（跟官方其它数据集一致）
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#
#         # if not self.nonautoregressive:
#         #     # 变成 token-supervision 形式（跟你仓库里其它 multivariate 数据集一致）
#         #     # seq_y: [L, C] -> unfold -> [N_tokens, C, P] -> reshape -> [N_tokens*C, P]
#         #     seq_y_t = torch.tensor(seq_y, dtype=torch.float32)
#         #     seq_y_t = seq_y_t.unfold(
#         #         dimension=0, size=self.output_token_len, step=self.input_token_len
#         #     ).permute(0, 2, 1).reshape(-1, self.output_token_len)
#         #     seq_y = seq_y_t
#         seq_x = torch.tensor(seq_x, dtype=torch.float32)  # shape: [96, 8]
#         seq_y = torch.tensor(seq_y, dtype=torch.float32)  # shape: [24, 8]
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def inverse_transform(self, data):
#         if not self.scale:
#             return data
#         x = np.asarray(data)
#         orig_shape = x.shape
#         x2 = x.reshape(-1, orig_shape[-1])
#         x2 = self.scaler.inverse_transform(x2)
#         return x2.reshape(orig_shape)
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# class PHM_MergedMultivariateNpy(Dataset):
#     """
#     PHM 走刀级 npy：每个 run 一个文件 {c1,c4,c6}.npy, shape=(315,8) (7ch + wear)
#     - train/val: 使用 train_runs
#     - test: 使用 test_runs
#     - train/val 在每个 run 内按窗口起点顺序做时间切分：train 前 split_ratio，val 后 1-split_ratio，中间留 gap_s
#     """
#
#     def __init__(
#         self,
#         root_path,
#         flag="train",
#         size=None,
#         data_path=None,
#         scale=True,
#         nonautoregressive=True,
#         subset_rand_ratio=1.0,
#
#         # ===== 交叉验证关键参数 =====
#         train_runs=("c1", "c4"),
#         test_runs=("c6",),
#
#         # ===== 时间切分参数 =====
#         split_ratio=0.9,     # 训练占比（在 train_runs 内）
#         time_gap=4,          # gap（单位：窗口起点 s 的间隔）
#         split_seed=2026,
#
#         wear_col=-1,
#         **kwargs,            # 吞掉 data_factory 可能额外传的参数（如 test_flag）
#     ):
#         assert flag in ["train", "val", "test"]
#         assert size is not None and len(size) == 3
#
#         self.root_path = root_path
#         self.flag = flag
#         self.seq_len = int(size[0])
#         self.input_token_len = int(size[1])
#         self.output_token_len = int(size[2])
#
#         self.data_path = data_path
#         self.scale = bool(scale)
#         self.nonautoregressive = bool(nonautoregressive)
#         self.subset_rand_ratio = float(subset_rand_ratio)
#
#         self.train_runs = list(train_runs) if isinstance(train_runs, (list, tuple)) else str(train_runs).split(",")
#         self.test_runs = list(test_runs) if isinstance(test_runs, (list, tuple)) else str(test_runs).split(",")
#
#         self.split_ratio = float(split_ratio)
#         self.time_gap = int(time_gap)
#         self.split_seed = int(split_seed)
#         self.wear_col = int(wear_col)
#
#         # 当前 dataset 需要的 runs
#         if self.flag in ["train", "val"]:
#             self.runs = list(self.train_runs)
#         else:
#             self.runs = list(self.test_runs)
#
#         self._read_data_and_build_index()
#
#     # --------------------------
#     # IO
#     # --------------------------
#     def _run_file(self, run_name: str) -> str:
#         """
#         兼容 data_path：
#         - data_path=None -> root_path/{run}.npy
#         - data_path 是目录 -> root_path/data_path/{run}.npy
#         - data_path 是文件名(如 c1.npy) -> 忽略，仍用 root_path/{run}.npy
#         """
#         default_path = os.path.join(self.root_path, f"{run_name}.npy")
#         if self.data_path is None:
#             return default_path
#
#         dp = str(self.data_path)
#         dp_abs = os.path.join(self.root_path, dp)
#
#         if os.path.isdir(dp_abs):
#             return os.path.join(dp_abs, f"{run_name}.npy")
#         if dp.endswith(".npy") or dp.endswith(".csv"):
#             return default_path
#         return default_path
#
#     def _read_one_run(self, run_name: str) -> np.ndarray:
#         f = self._run_file(run_name)
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Not found: {f}")
#         arr = np.load(f).astype(np.float32)
#         if arr.ndim != 2 or arr.shape[1] != 8:
#             raise ValueError(f"{f} must be (T,8), got {arr.shape}")
#         return arr
#
#     def _num_windows(self, T: int) -> int:
#         return T - self.seq_len - self.output_token_len + 1
#
#     # --------------------------
#     # Scaler (fit ONLY on train_runs' train coverage)
#     # --------------------------
#     def _fit_scaler_on_train_coverage(self, raw_runs: dict) -> StandardScaler:
#         scaler = StandardScaler()
#         chunks = []
#         for run in self.train_runs:
#             arr = raw_runs[run]
#             T = arr.shape[0]
#             n = self._num_windows(T)
#             if n <= 0:
#                 raise ValueError(f"Run {run}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}")
#
#             n_train = int(np.floor(n * self.split_ratio))
#             n_train = max(n_train, 1)
#
#             last_start = n_train - 1
#             max_end = min(last_start + self.seq_len + self.output_token_len, T)
#             chunks.append(arr[:max_end])
#
#         fit_mat = np.concatenate(chunks, axis=0)
#         scaler.fit(fit_mat)
#         return scaler
#
#     # --------------------------
#     # Build index
#     # --------------------------
#     def _read_data_and_build_index(self):
#         # 需要加载的 runs：当前 runs + train_runs（为了 fit scaler），以及 test_runs（若你想统一检查也可）
#         need = sorted(set(self.runs + self.train_runs + self.test_runs))
#         raw_runs = {r: self._read_one_run(r) for r in need}
#
#         self.scaler = None
#         if self.scale:
#             self.scaler = self._fit_scaler_on_train_coverage(raw_runs)
#
#         # transform 当前 dataset runs
#         self.data_runs = {}
#         for r in self.runs:
#             arr = raw_runs[r]
#             self.data_runs[r] = self.scaler.transform(arr) if self.scale else arr
#
#         # build index (run, start)
#         self.index = []
#         for r in self.runs:
#             data = self.data_runs[r]
#             T = data.shape[0]
#             n = self._num_windows(T)
#             if n <= 0:
#                 raise ValueError(f"Run {r}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}")
#
#             if self.flag in ["train", "val"]:
#                 n_train = int(np.floor(n * self.split_ratio))
#                 n_train = max(n_train, 1)
#
#                 gap_s = int(self.time_gap)
#                 if n_train + gap_s >= n:
#                     gap_s = 0
#
#                 starts = range(0, n_train) if self.flag == "train" else range(n_train + gap_s, n)
#             else:
#                 starts = range(0, n)
#
#             for s in starts:
#                 self.index.append((r, s))
#
#         # optional subset sampling
#         if self.flag == "train" and self.subset_rand_ratio < 1.0:
#             keep = max(int(len(self.index) * self.subset_rand_ratio), 1)
#             rng = np.random.default_rng(self.split_seed)
#             chosen = np.sort(rng.choice(len(self.index), size=keep, replace=False))
#             self.index = [self.index[i] for i in chosen]
#
#         # compatibility for exp_forecast.test()
#         self.raw = {f"{r}.npy": self.data_runs[r] for r in self.runs}
#         self.index_map = [(f"{r}.npy", s) for (r, s) in self.index]
#
#     # --------------------------
#     # PyTorch API
#     # --------------------------
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, idx):
#         run, s_begin = self.index[idx]
#         data = self.data_runs[run]
#         s_end = s_begin + self.seq_len
#
#         if self.nonautoregressive:
#             r_begin = s_end
#             r_end = r_begin + self.output_token_len
#             seq_x = torch.tensor(data[s_begin:s_end], dtype=torch.float32)   # [seq_len,8]
#             seq_y = torch.tensor(data[r_begin:r_end], dtype=torch.float32)   # [H,8]
#             seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#             seq_y_mark = torch.zeros((seq_y.shape[0], 1), dtype=torch.float32)
#             return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#         # unfold/autoregressive (保留接口)
#         r_begin = s_begin + self.input_token_len
#         r_end = s_end + self.output_token_len
#         seq_x = torch.tensor(data[s_begin:s_end], dtype=torch.float32)
#         seq_y_raw = torch.tensor(data[r_begin:r_end], dtype=torch.float32)
#
#         seq_y = seq_y_raw.unfold(dimension=0, size=self.output_token_len, step=self.input_token_len)  # [N,H,C]
#         seq_y = seq_y.permute(0, 2, 1).reshape(seq_y.shape[0] * seq_y.shape[1], -1)                  # [N*C,H]
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def inverse_transform(self, data):
#         if self.scaler is None:
#             return data
#         is_torch = torch.is_tensor(data)
#         x = data.detach().cpu().numpy() if is_torch else np.asarray(data)
#
#         mean = self.scaler.mean_
#         scale = self.scaler.scale_
#
#         if x.ndim == 1 or (x.ndim >= 2 and x.shape[-1] == 1):
#             inv = x * scale[self.wear_col] + mean[self.wear_col]
#             return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv
#
#         if x.shape[-1] != 8:
#             return data
#
#         x2 = x.reshape(-1, 8)
#         inv2 = x2 * scale.reshape(1, 8) + mean.reshape(1, 8)
#         inv = inv2.reshape(x.shape)
#         return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv

# 这里是硬编码的
# class PHM_MergedMultivariateNpy(Dataset):
#     """
#     PHM 走刀级数据（已聚合成 npy）专用：
#       - train/val: 用 c1 + c4
#       - test: 用 c6
#     每个 npy: shape = (315, 8)  # 7通道 + wear(最后一列)
#
#     size = [seq_len, input_token_len, output_token_len]
#       - nonautoregressive=True: 预测未来 output_token_len 步
#       - nonautoregressive=False: 保持和 Timer-XL 仓库里其它 Dataset 的 unfold 逻辑一致（可选）
#
#     切分策略（你确认的版本）：
#       - 每个 run 的窗口总数 n = T - seq_len - H + 1
#       - train: 前 split_ratio 的窗口起点（默认 0.9）
#       - val:  后 (1-split_ratio) 的窗口起点，且从 n_train + gap_s 开始（默认 gap_s=4）
#       - test: 用全部窗口（c6）
#     """
#
#     def __init__(
#         self,
#         root_path,
#         flag="train",
#         size=None,
#         data_path=None,              # 兼容接口（当前实现默认 root_path/{c1,c4,c6}.npy）
#         scale=True,
#         nonautoregressive=False,
#         test_flag="T",
#         subset_rand_ratio=1.0,
#
#         # ===== 时间切分参数（已按你要求设置默认）=====
#         split_ratio=0.9,            # train 前 90%，val 后 10%
#         time_gap=4,                 # gap（单位：窗口起点 s 的间隔），建议小一点
#         split_seed=2026,
#
#         wear_col=-1,                # wear 在最后一列
#     ):
#         assert flag in ["train", "val", "test"]
#         assert size is not None and len(size) == 3
#
#         self.root_path = root_path
#         self.flag = flag
#         self.seq_len = int(size[0])
#         self.input_token_len = int(size[1])
#         self.output_token_len = int(size[2])
#         self.scale = bool(scale)
#         self.nonautoregressive = bool(nonautoregressive)
#         self.test_flag = test_flag
#         self.subset_rand_ratio = float(subset_rand_ratio)
#
#         self.split_ratio = float(split_ratio)
#         self.time_gap = int(time_gap)
#         self.split_seed = int(split_seed)
#         self.wear_col = int(wear_col)
#
#         self.data_path = data_path
#
#         # runs
#         if self.flag in ["train", "val"]:
#             self.runs = ["c1", "c6"]
#         else:
#             self.runs = ["c4"]
#
#         self._read_data_and_build_index()
#
#     # --------------------------
#     # IO
#     # --------------------------
#     def _run_file(self, run_name: str) -> str:
#         """
#         兼容 data_path 传法：
#         1) data_path=None                 -> root_path/{run}.npy
#         2) data_path 是目录（存在）         -> root_path/data_path/{run}.npy
#         3) data_path 是文件名(如 c1.npy)   -> 忽略它，仍用 root_path/{run}.npy
#            （因为这是多 run 数据集，data_path 在很多脚本里会默认传一个文件名）
#         4) data_path 是绝对目录（存在）     -> data_path/{run}.npy
#         """
#         # 默认：每个 run 一个文件
#         default_path = os.path.join(self.root_path, f"{run_name}.npy")
#
#         if self.data_path is None:
#             return default_path
#
#         dp = str(self.data_path)
#
#         # 绝对路径且是目录
#         if os.path.isabs(dp) and os.path.isdir(dp):
#             return os.path.join(dp, f"{run_name}.npy")
#
#         # 相对路径：先拼到 root_path 下看看是不是目录
#         dp_abs = os.path.join(self.root_path, dp)
#         if os.path.isdir(dp_abs):
#             return os.path.join(dp_abs, f"{run_name}.npy")
#
#         # 如果传进来的是文件名（.npy/.csv 等），说明是别的数据集遗留参数 -> 忽略
#         if dp.endswith(".npy") or dp.endswith(".csv"):
#             return default_path
#
#         # 兜底：还是用默认路径
#         return default_path
#
#     def _read_one_run(self, run_name: str) -> np.ndarray:
#         f = self._run_file(run_name)
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Not found: {f}")
#         arr = np.load(f).astype(np.float32)
#         if arr.ndim != 2:
#             raise ValueError(f"{f} must be 2D, got shape={arr.shape}")
#         if arr.shape[1] != 8:
#             raise ValueError(f"{f} must have 8 cols (7ch+wear), got shape={arr.shape}")
#         return arr
#
#     def _num_windows(self, T: int) -> int:
#         # start s: x=[s, s+seq_len), y=[s+seq_len, s+seq_len+H)
#         return T - self.seq_len - self.output_token_len + 1
#
#     # --------------------------
#     # Scaler
#     # --------------------------
#     def _fit_scaler_on_train_coverage(self, raw_runs: dict) -> StandardScaler:
#         """
#         只在 c1+c4 的 train 覆盖到的时间范围内 fit scaler，避免把 val/test 后段信息带进去。
#         train windows: starts [0, n_train)
#         train 覆盖到的最大 y_end:
#           last_start = n_train-1
#           max_end = last_start + seq_len + H
#         """
#         scaler = StandardScaler()
#         chunks = []
#
#         for run in ["c1", "c4"]:
#             arr = raw_runs[run]
#             T = arr.shape[0]
#             n = self._num_windows(T)
#             if n <= 0:
#                 raise ValueError(f"Run {run}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}")
#
#             n_train = int(np.floor(n * self.split_ratio))
#             n_train = max(n_train, 1)
#
#             last_start = n_train - 1
#             max_end = last_start + self.seq_len + self.output_token_len
#             max_end = min(max_end, T)
#             chunks.append(arr[:max_end])
#
#         fit_mat = np.concatenate(chunks, axis=0)
#         scaler.fit(fit_mat)
#         return scaler
#
#     # --------------------------
#     # Build index
#     # --------------------------
#     def _read_data_and_build_index(self):
#         # 1) load raw runs
#         # scaler 永远用 c1+c4（即使当前 flag=test）
#         need = sorted(set(self.runs + ["c1", "c4"]))
#         raw_runs = {r: self._read_one_run(r) for r in need}
#
#         # 2) scaler
#         self.scaler = None
#         if self.scale:
#             self.scaler = self._fit_scaler_on_train_coverage(raw_runs)
#
#         # 3) transform current runs
#         self.data_runs = {}
#         self.raw_runs = {}
#         for r in self.runs:
#             arr = raw_runs[r]
#             self.raw_runs[r] = arr
#             if self.scale:
#                 self.data_runs[r] = self.scaler.transform(arr)
#             else:
#                 self.data_runs[r] = arr
#
#         # 4) build index
#         self.index = []
#         for r in self.runs:
#             data = self.data_runs[r]
#             T = data.shape[0]
#             n = self._num_windows(T)
#             if n <= 0:
#                 raise ValueError(
#                     f"Run {r}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}"
#                 )
#
#             if self.flag in ["train", "val"]:
#                 n_train = int(np.floor(n * self.split_ratio))
#                 n_train = max(n_train, 1)
#
#                 gap_s = int(self.time_gap)
#                 # 防止 gap 把 val 挤没：自动降到 0
#                 if n_train + gap_s >= n:
#                     gap_s = 0
#
#                 if self.flag == "train":
#                     starts = range(0, n_train)
#                 else:
#                     starts = range(n_train + gap_s, n)
#             else:
#                 starts = range(0, n)
#
#             for s in starts:
#                 self.index.append((r, s))
#
#         # 5) optional subset sampling for train
#         if self.flag == "train" and self.subset_rand_ratio < 1.0:
#             keep = max(int(len(self.index) * self.subset_rand_ratio), 1)
#             rng = np.random.default_rng(self.split_seed)
#             chosen = rng.choice(len(self.index), size=keep, replace=False)
#             chosen = np.sort(chosen)
#             self.index = [self.index[i] for i in chosen]
#
#         # 6) compatibility for exp_forecast.test()
#         self.raw = {f"{r}.npy": self.data_runs[r] for r in self.runs}
#         self.index_map = [(f"{r}.npy", s) for (r, s) in self.index]
#
#         print(f"[PHM][{self.flag}] runs={self.runs}  num_index={len(self.index)}  root={self.root_path}")
#
#     # --------------------------
#     # PyTorch Dataset API
#     # --------------------------
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, idx):
#         run, s_begin = self.index[idx]
#         data = self.data_runs[run]  # [T, 8]
#
#         s_end = s_begin + self.seq_len
#
#         if self.nonautoregressive:
#             # 未来 H 步
#             r_begin = s_end
#             r_end = r_begin + self.output_token_len
#
#             seq_x = data[s_begin:s_end]     # [seq_len, 8]
#             seq_y = data[r_begin:r_end]     # [H, 8]
#
#             seq_x = torch.tensor(seq_x, dtype=torch.float32)
#             seq_y = torch.tensor(seq_y, dtype=torch.float32)
#             seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#             seq_y_mark = torch.zeros((seq_y.shape[0], 1), dtype=torch.float32)
#             return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#         # ====== autoregressive/unfold 逻辑（保持你原来的风格）======
#         r_begin = s_begin + self.input_token_len
#         r_end = s_end + self.output_token_len
#
#         seq_x = data[s_begin:s_end]          # [seq_len, 8]
#         seq_y_raw = data[r_begin:r_end]      # [seq_len - input_token_len + H, 8]
#
#         seq_x = torch.tensor(seq_x, dtype=torch.float32)
#         seq_y = torch.tensor(seq_y_raw, dtype=torch.float32)
#
#         # unfold: [L, C] -> [N, H, C]
#         seq_y = seq_y.unfold(dimension=0, size=self.output_token_len, step=self.input_token_len)  # [N, H, C]
#         seq_y = seq_y.permute(0, 2, 1)  # [N, C, H]
#         seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)  # [N*C, H]
#
#         seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#         seq_y_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     # --------------------------
#     # Inverse transform
#     # --------------------------
#     def inverse_transform(self, data):
#         """
#         data: np.ndarray or torch.Tensor
#           - shape (..., 8) -> 全量逆变换
#           - shape (..., 1) 或 (...,) -> 当作 wear 逆变换（用 wear_col）
#         """
#         if self.scaler is None:
#             return data
#
#         is_torch = torch.is_tensor(data)
#         x = data.detach().cpu().numpy() if is_torch else np.asarray(data)
#
#         mean = self.scaler.mean_
#         scale = self.scaler.scale_
#
#         # wear-only
#         if x.ndim == 1:
#             inv = x * scale[self.wear_col] + mean[self.wear_col]
#             return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv
#
#         if x.shape[-1] == 1:
#             inv = x * scale[self.wear_col] + mean[self.wear_col]
#             return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv
#
#         # full (...,8)
#         if x.shape[-1] != 8:
#             return data
#
#         orig_shape = x.shape
#         x2 = x.reshape(-1, 8)
#         inv2 = x2 * scale.reshape(1, 8) + mean.reshape(1, 8)
#         inv = inv2.reshape(orig_shape)
#
#         return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv
# # 这里是133组特征的 查找
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PHM_MergedMultivariateNpy(Dataset):
    """
    走刀级 + X来自npz, wear来自 {run}_wear.csv
    data: [X, wear] -> last col is wear
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        data_path=None,
        scale=True,
        nonautoregressive=True,
        test_flag="T",
        keep_features_path=None,
        subset_rand_ratio=1.0,
        train_runs=("c1", "c4"),
        test_runs=("c6",),
        split_ratio=0.8,
        time_gap=0,
        split_seed=2026,
        wear_col=-1,
        file_suffix="_passlevel_full133.npz",
        wear_csv_dir=None,
        wear_csv_suffix="_wear.csv",
        wear_agg="max",                # "max" | "mean" | or a column name in csv
        mask_future_features_in_y=False,  # ✅ 可选：把 seq_y 里除wear外的特征置0，避免未来协变量被误用
        train_stride_candidates="1",
        train_stride_quantiles="",
        train_stride_use_monotonic_wear=True,
        train_stride_policy="slope",
        train_stride_random_seed=2026,
        train_window_weight_policy="none",
        train_window_weight_quantile=0.5,
        train_window_weight_seed=2026,
        **kwargs,
    ):
        assert flag in ["train", "val", "test"]
        assert size is not None and len(size) == 3

        self.root_path = root_path
        self.flag = flag
        self.seq_len = int(size[0])
        self.input_token_len = int(size[1])
        self.output_token_len = int(size[2])

        self.data_path = data_path
        self.file_suffix = str(file_suffix)

        self.scale = bool(scale)
        self.nonautoregressive = bool(nonautoregressive)
        self.test_flag = test_flag
        self.subset_rand_ratio = float(subset_rand_ratio)

        self.train_runs = list(train_runs) if isinstance(train_runs, (list, tuple)) else str(train_runs).split(",")
        self.test_runs = list(test_runs) if isinstance(test_runs, (list, tuple)) else str(test_runs).split(",")

        self.split_ratio = float(split_ratio)
        self.time_gap = int(time_gap)
        self.split_seed = int(split_seed)

        self.wear_col = int(wear_col)
        self.wear_csv_dir = wear_csv_dir
        self.wear_csv_suffix = wear_csv_suffix
        self.wear_agg = wear_agg
        self.mask_future_features_in_y = bool(mask_future_features_in_y)
        self.train_stride_candidates = self._parse_int_list(train_stride_candidates, [1])
        if 1 not in self.train_stride_candidates:
            self.train_stride_candidates = [1] + self.train_stride_candidates
        self.train_stride_candidates = sorted(set([s for s in self.train_stride_candidates if s >= 1]))
        self.train_stride_quantiles = self._parse_float_list(train_stride_quantiles, [])
        self.train_stride_use_monotonic_wear = bool(train_stride_use_monotonic_wear)
        self.train_stride_policy = str(train_stride_policy).strip().lower()
        if self.train_stride_policy not in {"slope", "random"}:
            print(f"[PHM-A2] unknown train_stride_policy={train_stride_policy}; fallback to slope")
            self.train_stride_policy = "slope"
        self.train_stride_random_seed = int(train_stride_random_seed)
        self.train_window_weight_policy = str(train_window_weight_policy).strip().lower()
        if self.train_window_weight_policy not in {"none", "stage_weight_only"}:
            print(f"[PHM-A2] unknown train_window_weight_policy={train_window_weight_policy}; fallback to none")
            self.train_window_weight_policy = "none"
        self.train_window_weight_quantile = float(train_window_weight_quantile)
        self.train_window_weight_seed = int(train_window_weight_seed)
        if any(s > 1 for s in self.train_stride_candidates) and (not self.nonautoregressive):
            print("[PHM-A2] stride augmentation only supports nonautoregressive mode; fallback to stride=1")
            self.train_stride_candidates = [1]

        self.runs = self.train_runs if self.flag in ["train", "val"] else self.test_runs

        # corr-prune
        self.keep_features_path = keep_features_path if keep_features_path else ""
        self.keep_feat_idx = None
        if self.keep_features_path.strip():
            names = []
            with open(self.keep_features_path, "r") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        names.append(s)

            idx = []
            for n in names:
                if n.startswith("Feat_"):
                    idx.append(int(n.split("_")[1]) - 1)
                else:
                    raise ValueError(f"Unknown feature name format in keep_features.txt: {n}")
            self.keep_feat_idx = sorted(idx)
            print(f"[CorrPrune] keep_features_path={self.keep_features_path} keep_X_cols={len(self.keep_feat_idx)}")

        self._read_data_and_build_index()

    @staticmethod
    def _parse_int_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        text = str(value).strip()
        if not text:
            return list(default)
        return [int(v.strip()) for v in text.split(",") if v.strip()]

    @staticmethod
    def _parse_float_list(value, default):
        if value is None:
            return list(default)
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).strip()
        if not text:
            return list(default)
        return [float(v.strip()) for v in text.split(",") if v.strip()]

    def _resolve_base_dir(self) -> str:
        base_dir = self.root_path
        if self.data_path is not None:
            dp = str(self.data_path)
            if os.path.isabs(dp) and os.path.isdir(dp):
                base_dir = dp
            else:
                dp_abs = os.path.join(self.root_path, dp)
                if os.path.isdir(dp_abs):
                    base_dir = dp_abs
        return base_dir

    def _run_file(self, run_name: str) -> str:
        base_dir = self._resolve_base_dir()
        p1 = os.path.join(base_dir, f"{run_name}{self.file_suffix}")
        p2 = os.path.join(base_dir, f"{run_name}.npz")
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"Cannot find npz for run={run_name}. Tried: {p1} and {p2}")

    def _read_wear_csv(self, run_name: str, T: int) -> np.ndarray:
        base_dir = self._resolve_base_dir()
        if self.wear_csv_dir is not None:
            wdir = str(self.wear_csv_dir)
            base_dir = wdir if os.path.isabs(wdir) else os.path.join(self.root_path, wdir)

        f = os.path.join(base_dir, f"{run_name}{self.wear_csv_suffix}")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Cannot find wear csv for run={run_name}: {f}")

        df = pd.read_csv(f)
        if "cut" in df.columns:
            df = df.sort_values("cut")

        if self.wear_agg in df.columns:
            y = df[self.wear_agg].to_numpy(dtype=np.float32)
        else:
            num_cols = [c for c in df.columns if c != "cut"]
            mat = df[num_cols].to_numpy(dtype=np.float32)
            if self.wear_agg == "mean":
                y = mat.mean(axis=1)
            else:
                y = mat.max(axis=1)

        if len(y) != T:
            raise ValueError(f"{f}: wear length={len(y)} but X length(T)={T}. Need aligned length.")
        return y.astype(np.float32)

    def _read_one_run(self, run_name: str) -> np.ndarray:
        f = self._run_file(run_name)
        d = np.load(f, allow_pickle=False)
        if "X" not in d.files:
            raise ValueError(f"{f} must contain key 'X'. Got keys={list(d.files)}")

        X = d["X"].astype(np.float32)
        if X.ndim != 2:
            raise ValueError(f"{f}: X must be 2D, got shape={X.shape}")
        T = X.shape[0]

        y = self._read_wear_csv(run_name, T)

        if self.keep_feat_idx is not None:
            X = X[:, self.keep_feat_idx]

        data = np.concatenate([X, y[:, None]], axis=1).astype(np.float32)

        if not hasattr(self, "raw_wear_um"):
            self.raw_wear_um = {}
        self.raw_wear_um[run_name] = y.copy()

        print(f"[Check] {run_name}: data shape={data.shape}")
        return data

    def _num_windows(self, T: int) -> int:
        return T - self.seq_len - self.output_token_len + 1

    def _window_fits(self, T: int, s_begin: int, stride: int) -> bool:
        last_x = s_begin + (self.seq_len - 1) * stride
        return (last_x + self.output_token_len) < T

    def _default_stride_quantiles(self, n_extra: int):
        if n_extra <= 0:
            return []
        if n_extra == 1:
            return [0.5]
        return np.linspace(0.5, 0.25, num=n_extra).tolist()

    def _prepare_train_stride_policy(self):
        self.train_hist_slope = {}
        self.train_stride_thresholds = {}
        self.train_random_allowed = {}
        self.train_stride_target_starts = {}

        extra_strides = [s for s in self.train_stride_candidates if s > 1]
        if self.flag != "train" or not extra_strides:
            return

        slope_values = []
        for run in self.train_runs:
            y = self.raw_wear_um[run].astype(np.float64)
            if self.train_stride_use_monotonic_wear:
                y = np.maximum.accumulate(y)

            T = len(y)
            n = self._num_windows(T)
            n_train = max(int(np.floor(n * self.split_ratio)), 1)

            run_slopes = {}
            for s_begin in range(0, n_train):
                last_hist = s_begin + self.seq_len - 1
                slope = float((y[last_hist] - y[s_begin]) / max(self.seq_len - 1, 1))
                run_slopes[s_begin] = slope
                slope_values.append(slope)
            self.train_hist_slope[run] = run_slopes

        if not slope_values:
            return

        quantiles = self.train_stride_quantiles
        if len(quantiles) != len(extra_strides):
            quantiles = self._default_stride_quantiles(len(extra_strides))

        for stride, q in zip(extra_strides, quantiles):
            self.train_stride_thresholds[stride] = float(np.quantile(slope_values, q))

        print(
            "[PHM-A2] train stride candidates={} thresholds={}".format(
                self.train_stride_candidates,
                {k: round(v, 6) for k, v in self.train_stride_thresholds.items()},
            )
        )

        for run in self.train_runs:
            y = self.raw_wear_um[run].astype(np.float64)
            T = len(y)
            n = self._num_windows(T)
            n_train = max(int(np.floor(n * self.split_ratio)), 1)
            per_run = {}
            for stride in extra_strides:
                thr = self.train_stride_thresholds.get(stride, None)
                if thr is None:
                    continue
                target_starts = []
                for s_begin in range(0, n_train):
                    if not self._window_fits(T, s_begin, stride):
                        continue
                    slope = self.train_hist_slope.get(run, {}).get(s_begin, None)
                    if slope is not None and slope <= thr:
                        target_starts.append(s_begin)
                per_run[stride] = target_starts
            self.train_stride_target_starts[run] = per_run

        if self.train_stride_policy == "random":
            rng = np.random.default_rng(self.train_stride_random_seed + self.split_seed)
            random_count_by_stride = {s: 0 for s in extra_strides}
            target_count_by_stride = {s: 0 for s in extra_strides}

            for run in self.train_runs:
                y = self.raw_wear_um[run].astype(np.float64)
                T = len(y)
                n = self._num_windows(T)
                n_train = max(int(np.floor(n * self.split_ratio)), 1)
                run_random = {}

                for stride in extra_strides:
                    thr = self.train_stride_thresholds.get(stride, None)
                    if thr is None:
                        continue

                    fit_starts = []
                    target_starts = list(self.train_stride_target_starts.get(run, {}).get(stride, []))
                    for s_begin in range(0, n_train):
                        if not self._window_fits(T, s_begin, stride):
                            continue
                        fit_starts.append(s_begin)

                    target_count = min(len(target_starts), len(fit_starts))
                    target_count_by_stride[stride] += target_count
                    if target_count <= 0:
                        continue

                    chosen = rng.choice(np.asarray(fit_starts, dtype=np.int64), size=target_count, replace=False)
                    for s_begin in chosen.tolist():
                        run_random.setdefault(int(s_begin), set()).add(stride)
                    random_count_by_stride[stride] += int(target_count)

                self.train_random_allowed[run] = run_random

            print(
                "[PHM-A2] random stride matching target_counts={} sampled_counts={}".format(
                    target_count_by_stride,
                    random_count_by_stride,
                )
            )

    def _prepare_stage_weight_policy(self):
        self.train_stage_weight_dup = {}
        self.train_stage_weight_meta = {}

        if self.flag != "train" or self.train_window_weight_policy != "stage_weight_only":
            return

        extra_strides = [s for s in self.train_stride_candidates if s > 1]
        if not extra_strides:
            print("[PHM-A2] stage_weight_only requested but no extra stride candidate exists; skip.")
            return

        ref_stride = min(extra_strides)
        rng = np.random.default_rng(self.train_window_weight_seed + self.split_seed)

        for run in self.train_runs:
            y = self.raw_wear_um[run].astype(np.float64)
            if self.train_stride_use_monotonic_wear:
                y = np.maximum.accumulate(y)

            T = len(y)
            n = self._num_windows(T)
            n_train = max(int(np.floor(n * self.split_ratio)), 1)
            target_count = len(self.train_stride_target_starts.get(run, {}).get(ref_stride, []))

            candidates = []
            for s_begin in range(0, n_train):
                if not self._window_fits(T, s_begin, 1):
                    continue
                last_hist = s_begin + self.seq_len - 1
                stage_score = float(y[last_hist])
                candidates.append((int(s_begin), stage_score))

            if target_count <= 0 or not candidates:
                self.train_stage_weight_dup[run] = {}
                self.train_stage_weight_meta[run] = {
                    "target_extra": int(target_count),
                    "selected_extra": 0,
                    "unique_selected": 0,
                    "score_threshold": None,
                }
                continue

            score_arr = np.asarray([score for _, score in candidates], dtype=np.float64)
            q = min(max(self.train_window_weight_quantile, 0.0), 1.0)
            score_threshold = float(np.quantile(score_arr, q))
            high_stage = [item for item in candidates if item[1] >= score_threshold]
            if not high_stage:
                high_stage = sorted(candidates, key=lambda x: x[1], reverse=True)

            pool_starts = np.asarray([s for s, _ in high_stage], dtype=np.int64)
            replace = target_count > len(pool_starts)
            chosen = rng.choice(pool_starts, size=target_count, replace=replace)

            dup_counts = {}
            for s_begin in chosen.tolist():
                dup_counts[s_begin] = dup_counts.get(int(s_begin), 0) + 1

            self.train_stage_weight_dup[run] = dup_counts
            self.train_stage_weight_meta[run] = {
                "target_extra": int(target_count),
                "selected_extra": int(sum(dup_counts.values())),
                "unique_selected": int(len(dup_counts)),
                "score_threshold": score_threshold,
            }

        print(
            "[PHM-A2] stage_weight_only ref_stride={} quantile={} meta={}".format(
                ref_stride,
                round(self.train_window_weight_quantile, 4),
                self.train_stage_weight_meta,
            )
        )

    def _allowed_train_strides(self, run: str, s_begin: int, T: int):
        allowed = [1]
        if self.flag != "train":
            return allowed

        if self.train_window_weight_policy == "stage_weight_only":
            return allowed

        if self.train_stride_policy == "random":
            extra = sorted(self.train_random_allowed.get(run, {}).get(s_begin, set()))
            return allowed + extra

        slope = self.train_hist_slope.get(run, {}).get(s_begin, None)
        if slope is None:
            return allowed

        for stride in self.train_stride_candidates:
            if stride == 1:
                continue
            threshold = self.train_stride_thresholds.get(stride, None)
            if threshold is None:
                continue
            if slope <= threshold and self._window_fits(T, s_begin, stride):
                allowed.append(stride)
        return allowed

    def _fit_scaler_on_train_coverage(self, raw_runs: dict) -> StandardScaler:
        scaler = StandardScaler()
        chunks = []
        for run in self.train_runs:
            arr = raw_runs[run]
            T = arr.shape[0]
            n = self._num_windows(T)
            if n <= 0:
                raise ValueError(f"Run {run}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}")

            n_train = int(np.floor(n * self.split_ratio))
            n_train = max(n_train, 1)

            last_start = n_train - 1
            max_end = min(last_start + self.seq_len + self.output_token_len, T)
            chunks.append(arr[:max_end])

        fit_mat = np.concatenate(chunks, axis=0)
        scaler.fit(fit_mat)
        return scaler

    def _read_data_and_build_index(self):
        need = sorted(set(self.runs + self.train_runs))
        raw_runs = {r: self._read_one_run(r) for r in need}

        self.scaler = None
        if self.scale:
            self.scaler = self._fit_scaler_on_train_coverage(raw_runs)

        self._prepare_train_stride_policy()
        self._prepare_stage_weight_policy()

        self.data_runs = {}
        for r in self.runs:
            arr = raw_runs[r]
            self.data_runs[r] = self.scaler.transform(arr) if self.scale else arr

        self.index = []
        for r in self.runs:
            data = self.data_runs[r]
            T = data.shape[0]
            n = self._num_windows(T)
            if n <= 0:
                raise ValueError(f"Run {r}: T={T} too short for seq_len={self.seq_len}, H={self.output_token_len}")

            if self.flag in ["train", "val"]:
                n_train = int(np.floor(n * self.split_ratio))
                n_train = max(n_train, 1)

                gap_s = int(self.time_gap)
                if n_train + gap_s >= n:
                    gap_s = 0

                starts = range(0, n_train) if self.flag == "train" else range(n_train + gap_s, n)
            else:
                starts = range(0, n)

            for s in starts:
                if self.flag == "train":
                    for stride in self._allowed_train_strides(r, s, T):
                        self.index.append((r, s, stride))
                else:
                    self.index.append((r, s, 1))

        if self.flag == "train" and self.train_window_weight_policy == "stage_weight_only":
            extra_index = []
            for r in self.runs:
                for s_begin, rep in sorted(self.train_stage_weight_dup.get(r, {}).items()):
                    extra_index.extend([(r, s_begin, 1)] * int(rep))
            self.index.extend(extra_index)
            print(f"[PHM-A2] stage_weight_only extra_train_windows={len(extra_index)}")

        if self.flag == "train" and self.subset_rand_ratio < 1.0:
            keep = max(int(len(self.index) * self.subset_rand_ratio), 1)
            rng = np.random.default_rng(self.split_seed)
            chosen = np.sort(rng.choice(len(self.index), size=keep, replace=False))
            self.index = [self.index[i] for i in chosen]

        self.raw = {f"{r}.npz": self.data_runs[r] for r in self.runs}
        self.index_map = [(f"{r}.npz", s, stride) for (r, s, stride) in self.index]

        if self.flag == "train":
            stride_stats = {}
            for _, _, stride in self.index:
                stride_stats[stride] = stride_stats.get(stride, 0) + 1
            print(f"[PHM-A2] train stride usage={stride_stats}")

        print(f"[PHM-B2][{self.flag}] runs={self.runs}  num_index={len(self.index)}  root={self.root_path}  data_path={self.data_path}")
        sample_run = self.runs[0]
        print(f"[Check] run={sample_run} data shape={self.data_runs[sample_run].shape}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        run, s_begin, stride = self.index[idx]
        data = self.data_runs[run]  # [T, C]

        if self.nonautoregressive:
            x_idx = s_begin + np.arange(self.seq_len) * stride
            last_x = int(x_idx[-1])
            y_idx = np.arange(last_x + 1, last_x + 1 + self.output_token_len)

            seq_x = torch.tensor(data[x_idx], dtype=torch.float32)   # [seq_len, C]
            seq_y = torch.tensor(data[y_idx], dtype=torch.float32)   # [H, C]

            # ✅ 可选：把未来段的特征列mask，只保留wear列（最后一列）
            if self.mask_future_features_in_y and seq_y.numel() > 0:
                seq_y[:, :-1] = 0.0

            seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
            seq_y_mark = torch.zeros((seq_y.shape[0], 1), dtype=torch.float32)
            return seq_x, seq_y, seq_x_mark, seq_y_mark

        # unfold/autoregressive
        if stride != 1:
            raise NotImplementedError("Stride-augmented PHM windows currently support nonautoregressive mode only.")

        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = torch.tensor(data[s_begin:s_end], dtype=torch.float32)
        seq_y_raw = torch.tensor(data[r_begin:r_end], dtype=torch.float32)

        seq_y = seq_y_raw.unfold(dimension=0, size=self.output_token_len, step=self.input_token_len)
        seq_y = seq_y.permute(0, 2, 1).reshape(seq_y.shape[0] * seq_y.shape[1], -1)

        seq_x_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
        seq_y_mark = torch.zeros((seq_x.shape[0], 1), dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        is_torch = torch.is_tensor(data)
        x = data.detach().cpu().numpy() if is_torch else np.asarray(data)

        mean = self.scaler.mean_
        scale = self.scaler.scale_

        if x.ndim == 1 or (x.ndim >= 1 and x.shape[-1] == 1):
            inv = x * scale[self.wear_col] + mean[self.wear_col]
            return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv

        C = len(mean)
        if x.shape[-1] != C:
            return data

        x2 = x.reshape(-1, C)
        inv2 = x2 * scale.reshape(1, C) + mean.reshape(1, C)
        inv = inv2.reshape(x.shape)
        return torch.tensor(inv, dtype=data.dtype, device=data.device) if is_torch else inv



class UnivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        feat_id = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
                
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
            
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Global_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "temp_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Global_Wind(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "wind_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Dataset_ERA5_Pretrain(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        Train_S = int(S * 0.8)
        df_raw = df_raw[:, :Train_S]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ERA5_Pretrain_Test(Dataset):
    def __init__(self, root_path, flag='test', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.test_flag = test_flag
        assert test_flag in ['T', 'V', 'TandV']
        type_map = {'T': 0, 'V': 1, 'TandV': 2}
        self.test_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        if self.test_type == 0:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, :Train_S]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len,
                        len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            data = df_raw
            border1 = border1s[-1]
            border2 = border2s[-1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        else:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, Train_S:]
            num_train = int(len(df_raw) * 0.8)
            num_test = len(df_raw) - num_train
            border1s = [0, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, len(df_raw)]
            data = df_raw
            if self.test_type == 1:
                border1 = border1s[0]
                border2 = border2s[0]
            else:
                border1 = border1s[1]
                border2 = border2s[1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Download link: https://huggingface.co/datasets/thuml/UTSD
class UTSD(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.root_path = root_path
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.csv'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    df_raw = pd.read_csv(dataset_path)

                    if isinstance(df_raw[df_raw.columns[0]][0], str):
                        data = df_raw[df_raw.columns[1:]].values
                    else:
                        data = df_raw.values

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)
                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]


# Download link: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
class UTSD_Npy(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    data = np.load(dataset_path)

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]
