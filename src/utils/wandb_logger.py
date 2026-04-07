import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import torch


class WandbLogger:
    """Small utility wrapper that keeps W&B optional and failure-safe."""

    def __init__(self, cfg: Dict[str, Any], save_dir: str, component: str):
        self.cfg = cfg or {}
        self.save_dir = save_dir
        self.component = component
        self.enabled = False
        self._wandb = None
        self._run = None
        self._system_step = -1
        self._system_log_every = int(
            self.cfg.get("wandb", {}).get("system_log_every_steps", 25)
        )

        wandb_cfg = self.cfg.get("wandb", {})
        if not wandb_cfg.get("enable", False):
            return

        try:
            import wandb  # type: ignore

            self._wandb = wandb
        except Exception as e:
            print(f"[W&B] disabled: import failed: {e}")
            return

        mode = "offline" if wandb_cfg.get("offline", False) else "online"
        scene = self.cfg.get("scene", "unknown_scene")
        dataset = self.cfg.get("dataset", "unknown_dataset")
        run_id = wandb_cfg.get("run_id", None)
        default_run_name = os.path.basename(os.path.normpath(save_dir)) or f"{dataset}-{scene}"
        run_name = wandb_cfg.get("run_name") or default_run_name
        tags = list(wandb_cfg.get("tags", []))
        tags.append(component)

        run_cfg = {
            "dataset": dataset,
            "scene": scene,
            "component": component,
            "save_dir": save_dir,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_cfg.update(self._sanitize_config(self.cfg))

        try:
            self._run = self._wandb.init(
                project=wandb_cfg.get("project", "wildgs-slam"),
                entity=wandb_cfg.get("entity", None),
                group=wandb_cfg.get("group", f"{dataset}-{scene}"),
                name=run_name,
                id=run_id,
                resume="allow",
                mode=mode,
                dir=save_dir,
                tags=tags,
                config=run_cfg,
                reinit=True,
            )
            self.enabled = self._run is not None
        except Exception as e:
            print(f"[W&B] disabled: init failed: {e}")
            self.enabled = False
            self._run = None

        if self.enabled:
            self.log_system_info()

    @staticmethod
    def _sanitize_config(cfg: Any):
        try:
            json.dumps(cfg)
            return cfg
        except Exception:
            return str(cfg)

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            if step is None:
                self._wandb.log(data)
            else:
                self._wandb.log(data, step=int(step))
        except Exception as e:
            print(f"[W&B] log failed: {e}")

    def add_scalar(self, tag: str, scalar_value: Any, global_step: Optional[int] = None):
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.detach().cpu().item()
        self.log({tag: scalar_value}, step=global_step)

    def add_text(self, tag: str, text: str, global_step: Optional[int] = None):
        if not self.enabled:
            return
        try:
            self.log({tag: self._wandb.Html(f"<pre>{text}</pre>")}, step=global_step)
        except Exception:
            self.log({tag: text}, step=global_step)

    def log_table(self, key: str, columns: Iterable[str], rows: Iterable[Iterable[Any]]):
        if not self.enabled:
            return
        try:
            table = self._wandb.Table(columns=list(columns))
            for row in rows:
                table.add_data(*list(row))
            self.log({key: table})
        except Exception as e:
            print(f"[W&B] table log failed: {e}")

    def log_system_info(self):
        info = {
            "system/hostname": socket.gethostname(),
            "system/platform": platform.platform(),
            "system/python": sys.version.split()[0],
            "system/pytorch": torch.__version__,
            "system/pid": os.getpid(),
            "system/cuda_available": torch.cuda.is_available(),
            "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            info.update(
                {
                    "system/cuda_version": torch.version.cuda,
                    "system/cudnn_version": torch.backends.cudnn.version(),
                    "system/gpu_name": props.name,
                    "system/gpu_total_mem_gb": props.total_memory / (1024**3),
                }
            )
        self.log(info, step=0)

    def log_system_stats(self, step: Optional[int] = None, force: bool = False):
        if not self.enabled:
            return
        if step is not None and not force:
            if step - self._system_step < self._system_log_every:
                return
            self._system_step = step

        # Keep only W&B automatic system metrics; avoid duplicate manual gpu/* charts.
        return

    def log_timer_stats(self, timer_stats: Dict[str, Dict[str, float]], step: Optional[int] = None):
        flat = {}
        for name, stats in timer_stats.items():
            prefix = f"timing/{name}"
            flat[f"{prefix}/count"] = stats.get("count", 0)
            flat[f"{prefix}/total_s"] = stats.get("total", 0.0)
            flat[f"{prefix}/avg_s"] = stats.get("avg", 0.0)
            flat[f"{prefix}/fps"] = stats.get("fps", 0.0)
            flat[f"{prefix}/min_s"] = stats.get("min", 0.0)
            flat[f"{prefix}/max_s"] = stats.get("max", 0.0)
        if flat:
            self.log(flat, step=step)

    def finish(self):
        if not self.enabled:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
