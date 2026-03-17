import os
import time
import warnings
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import matplotlib.pyplot as plt
import os, time
import numpy as np
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    @staticmethod
    def _strip_prefix_if_present(state_dict, prefix):
        if not state_dict:
            return state_dict
        if not all(k.startswith(prefix) for k in state_dict.keys()):
            return state_dict
        plen = len(prefix)
        return {k[plen:]: v for k, v in state_dict.items()}

    @staticmethod
    def _count_shape_matches(state_dict, model_state):
        cnt = 0
        for k, v in state_dict.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                cnt += 1
        return cnt

    def _align_state_for_model(self, state_dict, model_state):
        # Try common wrapper prefixes and keep the best match.
        candidates = [state_dict]
        candidates.append(self._strip_prefix_if_present(state_dict, "module."))
        candidates.append(self._strip_prefix_if_present(state_dict, "model."))
        candidates.append(self._strip_prefix_if_present(self._strip_prefix_if_present(state_dict, "model."), "module."))
        candidates.append(self._strip_prefix_if_present(self._strip_prefix_if_present(state_dict, "module."), "model."))

        best = state_dict
        best_score = -1
        for c in candidates:
            score = self._count_shape_matches(c, model_state)
            if score > best_score:
                best_score = score
                best = c
        return best
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            # fallback to CPU when CUDA is unavailable in current runtime.
            self.device = self.args.gpu if torch.cuda.is_available() else 'cpu'
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu if torch.cuda.is_available() else 'cpu'
            model = model.to(self.device)
            
        if self.args.adaptation:
            # zml修改
            # model.load_state_dict(torch.load(self.args.pretrain_model_path))
            # state = torch.load(self.args.pretrain_model_path, map_location='cpu')
            # model.load_state_dict(state, strict=False)
            state = torch.load(self.args.pretrain_model_path, map_location='cpu')

            # 兼容：有的ckpt会包一层
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]

            base_model = model.module if hasattr(model, "module") else model
            model_state = base_model.state_dict()
            state = self._align_state_for_model(state, model_state)

            # 只加载：key存在且shape完全一致的参数
            filtered = {}
            skipped = []
            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    skipped.append(k)

            print(f"[Pretrain] matched={len(filtered)} skipped={len(skipped)}")
            # print("[Pretrain] skipped keys:", skipped)  # 想调试再打开

            base_model.load_state_dict(filtered, strict=False)
            # ---- Freeze backbone: train head only ----
            # ---- Freeze backbone: unfreeze head + last K blocks ----
            # ---- Freeze backbone: unfreeze head + last K blocks (with regex fallback) ----
            if getattr(self.args, "freeze_backbone", False):
                import re

                base = model.module if hasattr(model, "module") else model

                # 1) freeze all
                for p in base.parameters():
                    p.requires_grad = False

                # 2) unfreeze head/projection (STRICT, no "out")
                def is_head_param(name: str) -> bool:
                    n = name.lower()
                    return (
                            n.startswith("head.") or (".head." in n) or
                            # ("projection" in n) or ("predict" in n) or ("decoder" in n) or
                            (".fc." in n) or n.endswith(".fc.weight") or n.endswith(".fc.bias") or
                            ("out_proj" in n) or ("output_proj" in n)
                    )

                for name, p in base.named_parameters():
                    if is_head_param(name):
                        p.requires_grad = True

                # 3) unfreeze last K blocks
                K = int(getattr(self.args, "unfreeze_last_n", 1))

                blocks = None
                found_path = None
                # 尝试常见容器路径（找得到就用）
                for path in [
                    "backbone.layers", "backbone.blocks", "backbone.h",
                    "encoder.layers", "transformer.h",
                    "model.layers", "layers", "blocks", "h",
                ]:
                    obj = base
                    ok = True
                    for part in path.split("."):
                        if not hasattr(obj, part):
                            ok = False
                            break
                        obj = getattr(obj, part)
                    if ok and hasattr(obj, "__len__"):
                        blocks = obj
                        found_path = path
                        break

                if blocks is not None:
                    L = len(blocks)
                    start = max(L - K, 0)
                    for i in range(start, L):
                        for p in blocks[i].parameters():
                            p.requires_grad = True
                    print(f"[Freeze] unfreeze blocks: last {K}/{L} from path={found_path}")
                else:
                    # ===== C) 你问的 fallback 就在这里：按层号 regex 解冻最后 K 层 =====
                    idxs = []
                    for n, _ in base.named_parameters():
                        for pat in [r"layers\.(\d+)\.", r"blocks\.(\d+)\.", r"h\.(\d+)\."]:
                            m = re.search(pat, n)
                            if m:
                                idxs.append(int(m.group(1)))

                    if idxs:
                        max_i = max(idxs)
                        target = set(range(max_i - K + 1, max_i + 1))
                        for n, p in base.named_parameters():
                            for pat in [r"layers\.(\d+)\.", r"blocks\.(\d+)\.", r"h\.(\d+)\."]:
                                m = re.search(pat, n)
                                if m and int(m.group(1)) in target:
                                    p.requires_grad = True
                        print(f"[Freeze] fallback unfreeze last {K} blocks by regex, max_layer_id={max_i}")
                    else:
                        print(
                            "[Freeze] WARNING: cannot find blocks container nor layer indices; only head/projection unfrozen.")

                trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
                total = sum(p.numel() for p in base.parameters())
                print(f"[Freeze] trainable params: {trainable}/{total} ({trainable / total:.2%})")

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _parse_run_cfg(self, run_cfg):
        if isinstance(run_cfg, (list, tuple)):
            vals = run_cfg
        else:
            vals = str(run_cfg).split(",")

        runs = []
        for v in vals:
            s = str(v).strip().lower()
            if s:
                runs.append(s)
        return runs

    def _build_train_bundle(self):
        # Non-PHM path keeps original single-loader behavior.
        if self.args.data != "PHM_MergedMultivariateNpy":
            train_data, train_loader = self._get_data(flag='train')
            return {
                "mode": "single",
                "train_data": train_data,
                "train_loader": train_loader,
                "loader_infos": [],
                "train_steps": len(train_loader),
            }

        enable_dual = bool(int(getattr(self.args, "enable_dual_loader", 1)))
        train_runs = self._parse_run_cfg(getattr(self.args, "train_runs", ""))

        if (not enable_dual) or (len(train_runs) < 2):
            train_data, train_loader = self._get_data(flag='train')
            reason = "disabled by enable_dual_loader=0" if (not enable_dual) else "train_runs < 2"
            print(f"[DualLoader] fallback to single loader: {reason}")
            return {
                "mode": "single",
                "train_data": train_data,
                "train_loader": train_loader,
                "loader_infos": [],
                "train_steps": len(train_loader),
            }

        # Build one merged-train dataset first, used as the shared scaler source.
        # This ensures c1/c4 per-run loaders in dual mode use the same normalization stats.
        merged_train_data, merged_train_loader = self._get_data(flag='train')
        shared_scaler = getattr(merged_train_data, "scaler", None)
        shared_steps = len(merged_train_loader)

        loader_infos = []
        total_steps = 0
        for run in train_runs:
            args_one = copy.deepcopy(self.args)
            args_one.train_runs = run
            ds, dl = data_provider(args_one, 'train')

            # Re-normalize per-run dataset with shared scaler from merged train runs.
            if (shared_scaler is not None) and hasattr(shared_scaler, "mean_") and hasattr(ds, "data_runs"):
                ds.scaler = copy.deepcopy(shared_scaler)
                for r in ds.runs:
                    raw_arr = ds._read_one_run(r)
                    ds.data_runs[r] = ds.scaler.transform(raw_arr)
                    if hasattr(ds, "raw"):
                        ds.raw[f"{r}.npz"] = ds.data_runs[r]

            steps = len(dl)
            loader_infos.append({
                "run": run,
                "data": ds,
                "loader": dl,
                "steps": steps,
            })
            total_steps += steps

        print(
            "[DualLoader] enabled | train_runs={} | per-run train steps={} | merged steps={}".format(
                [x["run"] for x in loader_infos],
                {x["run"]: x["steps"] for x in loader_infos},
                total_steps,
            )
        )
        if shared_scaler is not None and hasattr(shared_scaler, "mean_"):
            print(f"[DualLoader] shared scaler enabled from merged train runs, n_features={len(shared_scaler.mean_)}")
        print(f"[DualLoader] fair-compare shared train steps per epoch={shared_steps}")
        return {
            "mode": "dual",
            "train_data": merged_train_data,
            "train_loader": None,
            "loader_infos": loader_infos,
            "train_steps": shared_steps,
            "shared_steps": shared_steps,
        }

    def _iter_train_batches(self, train_bundle, epoch):
        if train_bundle["mode"] == "single":
            train_loader = train_bundle["train_loader"]
            if self.args.ddp:
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch + 1)
            for batch in train_loader:
                yield "merged", batch
            return

        infos = train_bundle["loader_infos"]
        if self.args.ddp:
            for info in infos:
                sampler = getattr(info["loader"], "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch + 1)
        shared_steps = int(train_bundle.get("shared_steps", train_bundle["train_steps"]))
        step_counts = np.array([max(int(info["steps"]), 1) for info in infos], dtype=np.float64)
        probs = step_counts / step_counts.sum()

        rng = np.random.default_rng(int(getattr(self.args, "seed", 2021)) + int(epoch))
        expected = probs * float(shared_steps)
        plan_counts = np.floor(expected).astype(np.int64)
        remain = int(shared_steps - int(plan_counts.sum()))
        if remain > 0:
            frac = expected - plan_counts
            order = np.argsort(-(frac + rng.random(len(infos)) * 1e-12))
            for idx in order[:remain]:
                plan_counts[idx] += 1
        elif remain < 0:
            order = np.argsort(-(plan_counts + rng.random(len(infos)) * 1e-12))
            for idx in order[:(-remain)]:
                if plan_counts[idx] > 0:
                    plan_counts[idx] -= 1

        plan = []
        for src_idx, cnt in enumerate(plan_counts.tolist()):
            plan.extend([src_idx] * int(max(cnt, 0)))
        if len(plan) < shared_steps:
            plan.extend([int(np.argmax(probs))] * (shared_steps - len(plan)))
        elif len(plan) > shared_steps:
            plan = plan[:shared_steps]
        plan = np.asarray(plan, dtype=np.int64)
        rng.shuffle(plan)
        iters = [iter(info["loader"]) for info in infos]

        for src_idx in plan:
            try:
                batch = next(iters[src_idx])
            except StopIteration:
                iters[src_idx] = iter(infos[src_idx]["loader"])
                batch = next(iters[src_idx])
            yield infos[src_idx]["run"], batch
    # zml
    # def _select_optimizer(self):
    #     p_list = []
    #     for n, p in self.model.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         else:
    #             p_list.append(p)
    #     model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    #     if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
    #         print('next learning rate is {}'.format(self.args.learning_rate))
    #     return model_optim
    def _select_optimizer(self):
        # 兼容 DDP/DP
        base = self.model.module if hasattr(self.model, "module") else self.model

        lr_head = float(self.args.learning_rate)
        lr_backbone = getattr(self.args, "learning_rate_backbone", None)
        lr_backbone = float(lr_head * 0.1) if (lr_backbone is None) else float(lr_backbone)

        def is_head_param(name: str) -> bool:
            n = name.lower()
            # 只解冻真正的 head / fc，不要用 projection 这种泛词
            return (
                    n.startswith("head.") or (".head." in n) or
                    (".fc." in n) or n.endswith(".fc.weight") or n.endswith(".fc.bias")
            )

        head_params, backbone_params = [], []
        for n, p in base.named_parameters():
            if not p.requires_grad:
                continue
            if is_head_param(n):
                head_params.append(p)
            else:
                backbone_params.append(p)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_backbone})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head})
        cnt = 0
        for n, p in base.named_parameters():
            if p.requires_grad:
                print("[Trainable]", n, p.numel())
                cnt += 1
                if cnt >= 30:
                    break
        if not param_groups:
            raise RuntimeError("No trainable params found. Check freeze/unfreeze logic.")

        model_optim = optim.AdamW(param_groups, weight_decay=self.args.weight_decay)

        if (self.args.ddp and self.args.local_rank == 0) or (not self.args.ddp):
            print(f"[Opt] lr_backbone={lr_backbone} (n={len(backbone_params)}), "
                  f"lr_head={lr_head} (n={len(head_params)})")

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        
        self.model.eval()    
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self, setting):
        train_bundle = self._build_train_bundle()
        train_data = train_bundle["train_data"]
        train_loader = train_bundle["train_loader"]
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = int(train_bundle["train_steps"])
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            source_batch_counter = {}
            self.model.train()
            epoch_time = time.time()
            for i, (source_tag, batch_pack) in enumerate(self._iter_train_batches(train_bundle, epoch)):
                source_batch_counter[source_tag] = source_batch_counter.get(source_tag, 0) + 1
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_pack
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                # loss = criterion(outputs, batch_y)
                # ===== base loss =====
                loss_mse = criterion(outputs, batch_y)

                # ===== monotonicity loss (penalize decreases) =====
                # outputs: [B, H]
                diff = outputs[:, 1:] - outputs[:, :-1]  # [B, H-1]
                loss_mono = torch.relu(-diff).mean()  # only penalize negative diffs

                # ===== smoothness loss (penalize "zig-zag") =====
                # second difference: y[t+1] - 2*y[t] + y[t-1]
                if outputs.size(1) >= 3:
                    d2 = outputs[:, 2:] - 2 * outputs[:, 1:-1] + outputs[:, :-2]  # [B, H-2]
                    loss_smooth = (d2 ** 2).mean()
                else:
                    loss_smooth = torch.tensor(0.0, device=outputs.device)

                # ===== weights (先用这组，比较稳) =====
                lam_mono = getattr(self.args, "lam_mono", 0.5)
                lam_smooth = getattr(self.args, "lam_smooth", 0.05)

                loss = loss_mse + lam_mono * loss_mono + lam_smooth * loss_smooth

                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                if train_bundle["mode"] == "dual":
                    print(f"[DualLoader][Epoch {epoch + 1}] batch_count_by_run={source_batch_counter}")

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model
    # 2026.1.20zml修改
    def test(self, setting, test=0):
        import os
        import time
        import math
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from utils.tools import visual

        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len,
              self.args.output_token_len, self.args.test_pred_len)

        # -------- load checkpoint (optional) --------
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            ckpt_path = os.path.join(self.args.checkpoints, setting, best_model_path)
            print("loading model from {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            # keep frozen params if missing in checkpoint
            for name, param in self.model.named_parameters():
                if (not param.requires_grad) and (name not in checkpoint):
                    checkpoint[name] = param

            self.model.load_state_dict(checkpoint, strict=False)

        # ============================================================
        # Roll-out inference (Leak-free, Scheme A)
        #   - only wear is rolled
        #   - future covariates are NOT used (filled by last-step copy)
        # ============================================================
        preds_wear_list = []
        trues_wear_list = []

        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0

        # horizon settings
        H = int(self.args.test_pred_len)
        chunk = int(self.args.output_token_len)
        steps = int(math.ceil(H / chunk))

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)  # (B, seq_len, Cx)
                batch_y = batch_y.float().to(self.device)  # (B, H(or more), C)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                B, seq_len, Cx = batch_x.shape

                # wear index in input channels (usually -1)
                widx = int(self.args.target_idx)
                widx = widx if widx >= 0 else (Cx + widx)
                if not (0 <= widx < Cx):
                    raise RuntimeError(f"Invalid target_idx: widx={widx} but input Cx={Cx}")

                # prepare rolling input
                x_roll = batch_x

                pred_chunks = []

                for s in range(steps):
                    out = self.model(x_roll, batch_x_mark, batch_y_mark)

                    # out can be:
                    #   (B,T)         wear-only
                    #   (B,T,1)       wear-only
                    #   (B,T,Cout)    multi-channel
                    if out.ndim == 2:
                        # (B,T) wear-only
                        pred_w = out[:, -chunk:]  # (B, chunk)
                    else:
                        out_last = out[:, -chunk:, :]  # (B, chunk, Cout)
                        Cout = out_last.shape[-1]
                        if Cout == 1:
                            pred_w = out_last[:, :, 0]  # (B, chunk)
                        else:
                            # if model outputs multi-channel, use same wear idx if valid
                            if widx >= Cout:
                                raise RuntimeError(
                                    f"Model output Cout={Cout} but wear index widx={widx} (input Cx={Cx}). "
                                    f"Your model seems not outputting full covariates."
                                )
                            pred_w = out_last[:, :, widx]  # (B, chunk)

                    pred_chunks.append(pred_w)

                    # ---- Scheme A update x_roll (NO future covariates) ----
                    # use last observed feature vector as placeholder for future covariates
                    last_step = x_roll[:, -1:, :].detach()  # (B,1,Cx)
                    new_steps = last_step.repeat(1, chunk, 1)  # (B,chunk,Cx)
                    new_steps[:, :, widx] = pred_w  # write predicted wear only

                    # keep length = seq_len
                    x_roll = torch.cat([x_roll[:, chunk:, :], new_steps], dim=1)

                pred_wear = torch.cat(pred_chunks, dim=1)[:, :H]  # (B, H)

                # true wear: always take last H steps from batch_y
                # batch_y may be (B,H,C) already (nonautoregressive dataset)
                if batch_y.ndim == 3:
                    Cy = batch_y.shape[-1]
                    tidx = int(self.args.target_idx)
                    tidx = tidx if tidx >= 0 else (Cy + tidx)
                    if not (0 <= tidx < Cy):
                        raise RuntimeError(f"Invalid target_idx for batch_y: tidx={tidx} but Cy={Cy}")
                    true_wear = batch_y[:, -H:, tidx]  # (B, H)
                elif batch_y.ndim == 2:
                    true_wear = batch_y[:, -H:]  # (B, H)
                else:
                    raise RuntimeError(f"Unexpected batch_y shape: {batch_y.shape}")

                preds_wear_list.append(pred_wear.detach().cpu())
                trues_wear_list.append(true_wear.detach().cpu())

                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or (not self.args.ddp):
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
                            i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                # optional visualize (window)
                if getattr(self.args, "visualize", False) and i % 2 == 0:
                    dir_path = os.path.join(folder_path, f'{H}/')
                    os.makedirs(dir_path, exist_ok=True)
                    gt = np.array(true_wear[0].detach().cpu())
                    pd = np.array(pred_wear[0].detach().cpu())
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

        # -------- concat (N, H) --------
        preds_wear = torch.cat(preds_wear_list, dim=0).numpy()
        trues_wear = torch.cat(trues_wear_list, dim=0).numpy()
        print("preds_wear shape:", preds_wear.shape, "trues_wear shape:", trues_wear.shape)

        # -------- inverse scale to um (wear only) --------
        preds_inv, trues_inv = preds_wear, trues_wear
        if hasattr(test_data, "scaler") and hasattr(test_data.scaler, "mean_") and hasattr(test_data.scaler, "scale_"):
            # scaler is fit on full channels; wear channel index should align with target_idx in dataset
            target_idx = int(self.args.target_idx)
            Csc = test_data.scaler.mean_.shape[0]
            target_idx = target_idx if target_idx >= 0 else (Csc + target_idx)

            mean_t = float(test_data.scaler.mean_[target_idx])
            scale_t = float(test_data.scaler.scale_[target_idx])

            preds_inv = preds_wear * scale_t + mean_t
            trues_inv = trues_wear * scale_t + mean_t

        print("preds_inv shape:", preds_inv.shape, "trues_inv shape:", trues_inv.shape)

        # =========================
        # (1) Window-level metrics
        # =========================
        err = preds_inv - trues_inv
        mse_win = float(np.mean(err ** 2))
        rmse_win = float(np.sqrt(mse_win))
        mae_win = float(np.mean(np.abs(err)))
        print(f"[Metric][window] mse(um^2):{mse_win}, rmse(um):{rmse_win}, mae(um):{mae_win}")

        # ========= plot one window =========
        sample_id = 0
        save_dir = os.path.join("./results", setting)
        os.makedirs(save_dir, exist_ok=True)

        plt.figure()
        plt.plot(trues_inv[sample_id], label=f"true({trues_inv.shape[1]})")
        plt.plot(preds_inv[sample_id], label=f"pred({preds_inv.shape[1]})")
        plt.title(f"PHM - one window forecast (id={sample_id})")
        plt.xlabel(f"horizon step (0..{preds_inv.shape[1] - 1})")
        plt.ylabel("wear (um)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"wear_window_{sample_id}.png"), dpi=200)
        plt.close()

        # ==========================================
        # (2) Full-curve reconstruction from windows
        # ==========================================
        index_map = test_data.index_map  # list of (fn, s_begin)
        seq_len_plot = int(getattr(self.args, "test_seq_len", getattr(self.args, "seq_len", 96)))
        H_plot = int(preds_inv.shape[1])

        fn0 = index_map[0][0]
        L = int(test_data.raw[fn0].shape[0])  # e.g. 315

        pred_bucket = [[] for _ in range(L)]
        true_bucket = [[] for _ in range(L)]

        for ii, (fn, s_begin) in enumerate(index_map):
            base_t = int(s_begin) + seq_len_plot
            for k in range(H_plot):
                t = base_t + k
                if 0 <= t < L:
                    pred_bucket[t].append(float(preds_inv[ii, k]))
                    true_bucket[t].append(float(trues_inv[ii, k]))

        pred_full = np.full((L,), np.nan, dtype=np.float32)
        true_full = np.full((L,), np.nan, dtype=np.float32)
        for t in range(L):
            if len(pred_bucket[t]) > 0:
                pred_full[t] = float(np.mean(pred_bucket[t]))
            if len(true_bucket[t]) > 0:
                true_full[t] = float(np.mean(true_bucket[t]))

        # =========================
        # (3) Full-curve metrics (mask NaN)
        # =========================
        mask = np.isfinite(pred_full) & np.isfinite(true_full)
        if np.any(mask):
            e2 = (pred_full[mask] - true_full[mask])
            mse_full = float(np.mean(e2 ** 2))
            rmse_full = float(np.sqrt(mse_full))
            mae_full = float(np.mean(np.abs(e2)))
            print(
                f"[Metric][fullcurve] mse(um^2):{mse_full}, rmse(um):{rmse_full}, mae(um):{mae_full}  (valid_points={int(mask.sum())}/{L})")
        else:
            print("[Metric][fullcurve] all NaN, cannot compute.")

        # ========= (B) true = raw wear (um), pred = windows (um) =========
        # Prefer raw_wear_um saved in dataset (from CSV), else fall back to true_full
        run0 = os.path.splitext(os.path.basename(fn0))[0]
        if hasattr(test_data, "raw_wear_um") and isinstance(test_data.raw_wear_um, dict) and (
                run0 in test_data.raw_wear_um):
            true_raw_full_um = test_data.raw_wear_um[run0].astype(np.float32)
            if len(true_raw_full_um) != L:
                raise RuntimeError(f"raw_wear_um[{run0}] length={len(true_raw_full_um)} != L={L}")
        else:
            true_raw_full_um = true_full.copy()

        valid = np.isfinite(pred_full) & np.isfinite(true_raw_full_um)
        valid = valid & (np.arange(L) >= seq_len_plot)
        if np.any(valid):
            err_full_raw = pred_full[valid] - true_raw_full_um[valid]
            mse_full_raw = float(np.mean(err_full_raw ** 2))
            rmse_full_raw = float(np.sqrt(mse_full_raw))
            mae_full_raw = float(np.mean(np.abs(err_full_raw)))
            print(
                f"[Metric][fullcurve_raw] mse(um^2):{mse_full_raw}, rmse(um):{rmse_full_raw}, mae(um):{mae_full_raw}  (valid_points={int(valid.sum())}/{L})")
        else:
            print("[Metric][fullcurve_raw] no valid points.")

        # ========= plot full curve =========
        plt.figure(figsize=(10, 4))
        plt.plot(true_raw_full_um, label="true(raw wear, um)")
        plt.plot(pred_full, label="pred(from windows, um)")
        plt.axvline(seq_len_plot, linestyle="--", label=f"forecast starts @ t={seq_len_plot}")
        plt.title("PHM - full wear curve (true=raw_um, pred=windows_um)")
        plt.xlabel(f"walk index (0..{L - 1})")
        plt.ylabel("wear (um)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "wear_full_curve_trueRaw_predWindows.png"), dpi=200)
        plt.close()

        return {
            "mse_win": mse_win, "rmse_win": rmse_win, "mae_win": mae_win,
            "pred_full": pred_full, "true_full": true_full,
            "true_raw_full_um": true_raw_full_um
        }

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     print("info:", self.args.test_seq_len, self.args.input_token_len,
    #         self.args.output_token_len, self.args.test_pred_len)
    #
    #     if test:
    #         print('loading model')
    #         setting = self.args.test_dir
    #         best_model_path = self.args.test_file_name
    #         ckpt_path = os.path.join(self.args.checkpoints, setting, best_model_path)
    #         print("loading model from {}".format(ckpt_path))
    #         checkpoint = torch.load(ckpt_path, map_location=self.device)
    #         for name, param in self.model.named_parameters():
    #             if not param.requires_grad and name not in checkpoint:
    #                 checkpoint[name] = param
    #         self.model.load_state_dict(checkpoint, strict=False)
    #
    #     preds = []
    #     trues = []
    #
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     time_now = time.time()
    #     test_steps = len(test_loader)
    #     iter_count = 0
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             iter_count += 1
    #             batch_x = batch_x.float().to(self.device)            # [B, L, C]
    #             batch_y = batch_y.float().to(self.device)            # [B, L2, C]
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # ---- how many rollout chunks ----
    #             inference_steps = self.args.test_pred_len // self.args.output_token_len
    #             dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
    #             if dis != 0:
    #                 inference_steps += 1
    #
    #             wear_idx = self.args.target_idx  # e.g., -1
    #             pred_wear_chunks = []
    #
    #             for _ in range(inference_steps):
    #                 outputs = self.model(batch_x, batch_x_mark, batch_y_mark)   # usually [B, L, C]
    #                 # 取最后 H 步的 wear 预测 -> [B, H]
    #                 pred_wear = outputs[:, -self.args.output_token_len:, wear_idx]
    #                 pred_wear_chunks.append(pred_wear)
    #
    #                 # roll: 复制最后一步特征，只替换 wear
    #                 last_step = batch_x[:, -1:, :].detach()  # [B,1,C]
    #                 new_steps = last_step.repeat(1, self.args.output_token_len, 1)  # [B,H,C]
    #                 new_steps[:, :, wear_idx] = pred_wear
    #
    #                 # 保持 seq_len 不变：丢掉最早 H 步，拼上新 H 步
    #                 batch_x = torch.cat([batch_x[:, self.args.output_token_len:, :], new_steps], dim=1)
    #
    #             # 拼成 [B, test_pred_len]
    #             # pred_wear = torch.cat(pred_wear_chunks, dim=1)
    #             pred_wear = pred_wear[:, :self.args.test_pred_len]
    #             if dis != 0:
    #                 pred_wear = pred_wear[:, :dis]
    #
    #             true_wear = batch_y[:, -self.args.test_pred_len:, wear_idx]  # [B, test_pred_len]
    #
    #             preds.append(pred_wear.detach().cpu())
    #             trues.append(true_wear.detach().cpu())
    #
    #             if (i + 1) % 100 == 0:
    #                 if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
    #                     speed = (time.time() - time_now) / iter_count
    #                     left_time = speed * (test_steps - i)
    #                     print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(
    #                         i + 1, speed, left_time))
    #                     iter_count = 0
    #                     time_now = time.time()
    #
    #     # ---- concat to numpy (N, T) ----
    #     preds_wear = torch.cat(preds, dim=0).numpy()
    #     trues_wear = torch.cat(trues, dim=0).numpy()
    #     print("preds_wear shape:", preds_wear.shape, "trues_wear shape:", trues_wear.shape)
    #
    #     # ---- inverse scale (wear only) ----
    #     if hasattr(test_data, "scaler") and hasattr(test_data.scaler, "mean_") and hasattr(test_data.scaler, "scale_"):
    #         target_idx = self.args.target_idx
    #         C = test_data.scaler.mean_.shape[0]
    #         target_idx = target_idx if target_idx >= 0 else (C + target_idx)
    #
    #         mean_t = test_data.scaler.mean_[target_idx]
    #         scale_t = test_data.scaler.scale_[target_idx]
    #
    #         preds_inv = preds_wear * scale_t + mean_t
    #         trues_inv = trues_wear * scale_t + mean_t
    #     else:
    #         preds_inv = preds_wear
    #         trues_inv = trues_wear
    #
    #     print("preds_inv shape:", preds_inv.shape, "trues_inv shape:", trues_inv.shape)
    #
    #     err = preds_inv - trues_inv
    #     mse = np.mean(err ** 2)
    #     rmse = np.sqrt(mse)
    #     mae = np.mean(np.abs(err))
    #     print(f"mse(um^2):{mse}, rmse(um):{rmse}, mae(um):{mae}")
    #
    #     # ========= (A) 画一个样本窗口 =========
    #     H = int(preds_inv.shape[1])
    #     sample_id = 0
    #     plt.figure()
    #     plt.plot(trues_inv[sample_id], label=f"true({H})")
    #     plt.plot(preds_inv[sample_id], label=f"pred({H})")
    #     plt.title(f"PHM - one window forecast (id={sample_id})")
    #     plt.xlabel(f"horizon step (0..{H-1})")
    #     plt.ylabel("wear (um)")
    #     plt.legend()
    #     save_dir = os.path.join("./results", setting)
    #     os.makedirs(save_dir, exist_ok=True)
    #     plt.savefig(os.path.join(save_dir, f"wear_window_{sample_id}.png"), dpi=200)
    #     plt.close()
    #
    #     # ========= (B) 拼整段曲线：从窗口拼 =========
    #     index_map = test_data.index_map
    #     # seq_len = int(getattr(self.args, "seq_len", 96))
    #     seq_len = int(getattr(self.args, "test_seq_len", getattr(self.args, "seq_len", test_data.seq_len)))
    #
    #     fn0 = index_map[0][0]
    #     L = test_data.raw[fn0].shape[0]  # e.g., 315
    #
    #     pred_bucket = [[] for _ in range(L)]
    #     true_bucket = [[] for _ in range(L)]
    #
    #     for i, (fn, s) in enumerate(index_map):
    #         base_t = s + seq_len
    #         for k in range(H):
    #             t = base_t + k
    #             if 0 <= t < L:
    #                 pred_bucket[t].append(float(preds_inv[i, k]))
    #                 true_bucket[t].append(float(trues_inv[i, k]))
    #
    #     pred_full = np.full((L,), np.nan, dtype=np.float32)
    #     true_full = np.full((L,), np.nan, dtype=np.float32)
    #     for t in range(L):
    #         if pred_bucket[t]:
    #             pred_full[t] = float(np.mean(pred_bucket[t]))
    #         if true_bucket[t]:
    #             true_full[t] = float(np.mean(true_bucket[t]))
    #
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(true_full, label="true(full from windows)")
    #     plt.plot(pred_full, label="pred(full from windows)")
    #     plt.axvline(seq_len, linestyle="--", label=f"forecast starts @ t={seq_len}")
    #     plt.title("PHM - full wear curve (from sliding windows)")
    #     plt.xlabel("pass index")
    #     plt.ylabel("wear (um)")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, "wear_full_curve_from_windows.png"), dpi=200)
    #     plt.close()
    #
    #     print("[Plot] saved:", os.path.join(save_dir, "wear_full_curve_from_windows.png"))
    #     return
