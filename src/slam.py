import os
import json
import ast
import uuid
from datetime import datetime
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval, full_traj_fill
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from torch.utils.tensorboard import SummaryWriter
from src.utils.sys_timer import timer
from src.utils.wandb_logger import WandbLogger
import ctypes


class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.cfg.setdefault("wandb", {})
        self.cfg["wandb"].setdefault("enable", False)
        self.cfg["wandb"].setdefault("offline", False)
        self.cfg["wandb"].setdefault("project", "wildgs-slam")
        self.cfg["wandb"].setdefault("entity", None)
        self.cfg["wandb"].setdefault("group", None)
        self.cfg["wandb"].setdefault("run_name", None)
        self.cfg["wandb"].setdefault("track_ate_per_keyframe", True)
        self.cfg["wandb"].setdefault("system_log_every_steps", 25)
        self.cfg["wandb"].setdefault("tags", [])
        if self.cfg["wandb"].get("run_id") is None:
            self.cfg["wandb"]["run_id"] = uuid.uuid4().hex

        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.wandb_logger = None
        self.raw = cfg.get("raw", False)
        self.use_mlp = cfg.get("use_mlp", False)

        save_dir_base = os.path.join(cfg["data"]["output"], cfg["scene"])
        if self.raw:
            if self.use_mlp:
                save_dir_base = os.path.join(save_dir_base, "raw_mlp")
            else:
                save_dir_base = os.path.join(save_dir_base, "raw")
        else:
            save_dir_base = os.path.join(save_dir_base, "srgb")

        save_dir = save_dir_base
        version = 2

        while os.path.exists(save_dir):
            save_dir = f"{save_dir_base}_{version}"
            version += 1
        if version > 0:
            version -= 1
        else:
            version = ""
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(len(stream))

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        self.video = DepthVideo(cfg, self.printer)
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream
        self.final_clean = False

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, pipe):
        for file in os.listdir(self.save_dir):
            if file.startswith("events.out.tfevents."):
                os.remove(os.path.join(self.save_dir, file))

        event_writer = SummaryWriter(self.save_dir)
        self.wandb_logger = WandbLogger(self.cfg, self.save_dir, component="tracking")
        self.logger = self.wandb_logger
        self.tracker = Tracker(self, pipe, event_writer)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        timer._report_summary(self.save_dir)
        if self.wandb_logger is not None:
            self.wandb_logger.log_timer_stats(timer.get_function_stats())
            self.wandb_logger.log_system_stats(force=True)
            self.wandb_logger.finish()
        event_writer.close()
        self.printer.print("Tracking Done!", FontColor.TRACKER)

        if not self.cfg["mapping"]["enable"]:
            self.terminate()

    def mapping(self, pipe, q_main2vis, q_vis2main):
        self.wandb_logger = WandbLogger(self.cfg, self.save_dir, component="mapping")
        self.logger = self.wandb_logger
        self.mapper = Mapper(self, pipe, q_main2vis, q_vis2main, self.save_dir)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        if self.cfg["mapping"]["enable"]:
            self.terminate()

        timer._report_summary(self.save_dir)
        if self.wandb_logger is not None:
            self.wandb_logger.log_timer_stats(timer.get_function_stats())
            self.wandb_logger.log_system_stats(force=True)
            self.wandb_logger.finish()

    @timer.section("Final Global BA")
    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'])
        torch.cuda.empty_cache()
        self.ba.dense_ba(12, enable_udba=self.cfg['tracking']['frontend']['enable_opt_dyn_mask'], save_edges_weights=False)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)
            if self.cfg["mapping"]["enable"]:
                self.mapper.save_all_kf_figs(
                    self.save_dir,
                    iteration="before_refine",
                )
            if self.cfg["tracking"]["uncertainty_params"]["visualize"]:
                self.video.visualize_all_opt_params(
                    self.save_dir,
                    iteration="final",
                )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if self.cfg["mapping"]["enable"]:
            if self.cfg["tracking"]["backend"]["final_ba"]:
                self.mapper.final_refine(iters=self.cfg["mapping"]["final_refine_iters"])

            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="after_refine",
            )

            self.traj_filler.setup_feature_extractor()
            traj_est = full_traj_fill(
                self.traj_filler,
                self.mapper,
                self.stream,
                fast_mode=self.cfg['fast_mode'],
            )
            full_traj_eval(
                traj_est,
                self.stream,
                self.printer,
                self.logger,
                f"{self.save_dir}/traj",
                "full_traj",
            )
            self._save_final_metrics_txt()
            self._log_wandb_final_metrics()

            self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        else:
            with timer.section("Full Trajectory Filling"):
                self.traj_filler.setup_feature_extractor()
                traj_est = full_traj_fill(
                    self.traj_filler,
                    None,
                    self.stream,
                    fast_mode=True,
                )
            full_traj_eval(
                traj_est,
                self.stream,
                self.printer,
                self.logger,
                f"{self.save_dir}/traj",
                "full_traj",
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)
        timer._report_summary(self.save_dir)
        self.final_clean = True

    def _load_json_if_exists(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.printer.print(f"Failed to parse {path}: {e}", FontColor.WARNING)
            return None

    def _parse_traj_metrics_txt(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                text = f.read()
        except Exception as e:
            self.printer.print(f"Failed to read {path}: {e}", FontColor.WARNING)
            return None

        stats_marker = "statistics:"
        marker_idx = text.find(stats_marker)
        if marker_idx == -1:
            return None

        stats_text = text[marker_idx + len(stats_marker):].strip()
        try:
            return ast.literal_eval(stats_text)
        except Exception as e:
            self.printer.print(
                f"Failed to parse trajectory stats from {path}: {e}",
                FontColor.WARNING,
            )
            return None

    def _log_wandb_final_metrics(self):
        if self.wandb_logger is None:
            return

        before_render = self._load_json_if_exists(
            os.path.join(self.save_dir, "plots_before_refine", "render_metrics_summary.json")
        )
        after_render = self._load_json_if_exists(
            os.path.join(self.save_dir, "plots_after_refine", "render_metrics_summary.json")
        )

        before_kf_ate = self._parse_traj_metrics_txt(
            os.path.join(self.save_dir, "traj", "before_final_ba", "metrics_kf_traj.txt")
        )
        after_kf_ate = self._parse_traj_metrics_txt(
            os.path.join(self.save_dir, "traj", "metrics_kf_traj.txt")
        )
        full_traj_ate = self._parse_traj_metrics_txt(
            os.path.join(self.save_dir, "traj", "metrics_full_traj.txt")
        )

        rows = []
        for stage, render, kf_ate in [
            ("before_refine", before_render, before_kf_ate),
            ("after_refine", after_render, after_kf_ate),
        ]:
            rows.append(
                [
                    stage,
                    (render or {}).get("mean_psnr"),
                    (render or {}).get("mean_ssim"),
                    (render or {}).get("mean_lpips"),
                    (render or {}).get("depth_l1"),
                    (render or {}).get("mean_psnr_raw"),
                    (render or {}).get("mean_ssim_raw"),
                    (kf_ate or {}).get("rmse"),
                    (kf_ate or {}).get("mean"),
                    (kf_ate or {}).get("median"),
                ]
            )

        self.wandb_logger.log_table(
            key="summary/before_after_refine_metrics",
            columns=[
                "stage",
                "mean_psnr",
                "mean_ssim",
                "mean_lpips",
                "depth_l1",
                "mean_psnr_raw",
                "mean_ssim_raw",
                "kf_ate_rmse",
                "kf_ate_mean",
                "kf_ate_median",
            ],
            rows=rows,
        )

        if full_traj_ate is not None:
            self.wandb_logger.log(
                {
                    "summary/full_traj_ate_rmse": full_traj_ate.get("rmse"),
                    "summary/full_traj_ate_mean": full_traj_ate.get("mean"),
                    "summary/full_traj_ate_median": full_traj_ate.get("median"),
                }
            )

    def _save_final_metrics_txt(self):
        traj_metrics_path = os.path.join(self.save_dir, "traj", "metrics_full_traj.txt")
        render_metrics_path = os.path.join(
            self.save_dir, "plots_after_refine", "render_metrics_summary.json"
        )
        out_path = os.path.join(self.save_dir, "final_metrics.txt")

        if not os.path.exists(traj_metrics_path):
            self.printer.print(
                f"Skip final_metrics.txt: missing {traj_metrics_path}",
                FontColor.WARNING,
            )
            return
        if not os.path.exists(render_metrics_path):
            self.printer.print(
                f"Skip final_metrics.txt: missing {render_metrics_path}",
                FontColor.WARNING,
            )
            return

        with open(traj_metrics_path, "r") as f:
            traj_text = f.read()

        rmse = None
        stats_marker = "statistics:"
        marker_idx = traj_text.find(stats_marker)
        if marker_idx != -1:
            stats_text = traj_text[marker_idx + len(stats_marker):].strip()
            try:
                stats_dict = ast.literal_eval(stats_text)
                rmse = stats_dict.get("rmse", None)
            except (SyntaxError, ValueError):
                rmse = None

        with open(render_metrics_path, "r") as f:
            render_metrics = json.load(f)

        psnr = render_metrics.get("mean_psnr", None)
        ssim = render_metrics.get("mean_ssim", None)
        lpips = render_metrics.get("mean_lpips", None)
        depth_l1 = render_metrics.get("depth_l1", None)

        header = "# ate rmse (m) | psnr | ssim | lpips | depth_l1 (m)"
        row_values = [rmse, psnr, ssim, lpips, depth_l1]
        row = " , ".join("null" if v is None else str(v) for v in row_values)

        with open(out_path, "w") as f:
            f.write(header + "\n")
            f.write(row + "\n")

        run_created_at_path = os.path.join(self.save_dir, "run_created_at.txt")
        run_created_at_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(run_created_at_path, "w") as f:
            f.write(f"run_created_at: {run_created_at_str}\n")

        self.printer.print(f"Saved final metrics to {out_path}", FontColor.EVAL)
        self.printer.print(f"Saved run timestamp to {run_created_at_path}", FontColor.EVAL)

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def run(self):
        mp.set_start_method("spawn", force=True)
        exit_event = mp.Event()

        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        if self.cfg['mapping']['enable']:
            processes = [
                mp.Process(target=self.tracking, args=(t_pipe,)),
                mp.Process(target=self.mapping, args=(m_pipe, q_main2vis, q_vis2main)),
            ]
        else:
            processes = [
                mp.Process(target=self.tracking, args=(t_pipe,)),
            ]
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        if self.cfg['gui']:
            time.sleep(5)
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
            gaussians = GaussianModel(
                self.cfg['mapping']['model_params']['sh_degree'],
                config=self.cfg,
                dataset=self.cfg.get("dataset"),
                raw=self.raw,
                use_mlp=self.use_mlp,
            )

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()

        if self.cfg['droidvis']:
            from src.utils.droid_visualization_rerun import droid_visualization_rerun

            self.visualizer = mp.Process(
                target=droid_visualization_rerun,
                args=(self.video,),
                kwargs=dict(
                    web_port=9876,
                    record_path=f"{self.save_dir}/rerun_stream.rrd",
                    exit_event=exit_event,
                ),
            )
            self.visualizer.start()

        for p in processes:
            p.join()

        if self.cfg['droidvis'] and self.visualizer.is_alive():
            exit_event.set()
            self.visualizer.join(timeout=10)

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
