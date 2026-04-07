from src.motion_filter import MotionFilter
from src.frontend import Frontend
from src.backend import Backend
import time
import torch
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer, FontColor
from src.utils.datasets import RGB_NoPose
from src.utils.eval_traj import kf_traj_eval, kf_traj_ate
from src.utils.sys_timer import timer
import os
from torch.utils.tensorboard import SummaryWriter


class Tracker:
    def __init__(self, slam, pipe: Connection, event_writer: SummaryWriter):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
        self.output = slam.save_dir
        self.event_writer = event_writer
        self.wandb_logger = getattr(slam, "wandb_logger", None)

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(
            self.net,
            self.video,
            self.cfg,
            thresh=filter_thresh,
            device=self.device,
            output=self.output,
        )
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)
        self.online_ba = Backend(self.net, self.video, self.cfg)
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']
        self.finish_first_online_ba = False

        self.printer: Printer = slam.printer

        self.prev_kf_counter = 0

        # write the config to the event writer
        self.event_writer.add_text("config", str(self.cfg))
        if self.wandb_logger is not None:
            self.wandb_logger.add_text("config", str(self.cfg), global_step=0)

    def _log_tracking_step(self, frame_idx: int, frame_dt: float):
        if self.wandb_logger is None:
            return

        stats = self.motion_filter.last_track_stats
        payload = {
            "tracking/frame_time_s": frame_dt,
            "tracking/frame_fps": (1.0 / frame_dt) if frame_dt > 0 else 0.0,
            "tracking/video_counter": int(self.video.counter.value),
            "tracking/is_initialized": int(self.frontend.is_initialized),
            "tracking/delta_mean": float(stats.get("delta_mean", 0.0)),
            "tracking/is_keyframe_added": int(bool(stats.get("is_keyframe", False))),
        }
        self.wandb_logger.log(payload, step=frame_idx)
        self.wandb_logger.log_system_stats(step=frame_idx)

    def run(self, stream: BaseDataset):
        '''
        Trigger the tracking process.
        1. check whether there is enough motion between the current frame and last keyframe by motion_filter
        2. use frontend to do local bundle adjustment, to estimate camera pose and depth image,
            also delete the current keyframe if it is too close to the previous keyframe after local BA.
        3. run online global BA periodically by backend
        4. send the estimated pose and depth to mapper,
            and wait until the mapper finish its current mapping optimization.
        '''
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

        intrinsic = stream.get_intrinsic()
        wandb_ate_cfg = self.cfg.get("wandb", {})
        log_ate_per_kf = bool(wandb_ate_cfg.get("track_ate_per_keyframe", True))

        for i in range(len(stream)):  # iterate through the stream
            frame_t0 = time.perf_counter()
            timestamp, image, _, _ = stream[i]
            with torch.no_grad():
                starting_count = self.video.counter.value

                ### check there is enough motion
                with timer.section("Tracking"):
                    force_to_add_keyframe = self.motion_filter.track(
                        timestamp, image, intrinsic
                    )
                    # local bundle adjustment
                    self.frontend(force_to_add_keyframe, self.event_writer)

                # visualize the affine weights
                if self.video.enable_affine_transform and self.cfg['debug']:
                    self.event_writer.add_scalar(
                        "affine_weights/max", self.video.affine_weights.max(), timestamp
                    )
                    self.event_writer.add_scalar(
                        "affine_weights/min", self.video.affine_weights.min(), timestamp
                    )
                    self.event_writer.add_scalar(
                        "affine_weights/mean", self.video.affine_weights.mean(), timestamp
                    )
                    self.event_writer.add_scalar(
                        "affine_weights/std", self.video.affine_weights.std(), timestamp
                    )
                    # visualize the affine weights for each feature every 50 features
                    for feat_idx in range(self.video.affine_weights.shape[0] // 50):
                        self.event_writer.add_scalar(
                            f"affine_weights/feat{feat_idx*50}",
                            self.video.affine_weights[feat_idx * 50],
                            timestamp,
                        )

                if (
                    self.cfg['tracking']["uncertainty_params"]['visualize']
                    and self.cfg['debug']
                ):
                    if (
                        not isinstance(stream, RGB_NoPose)
                        and self.video.counter.value > self.prev_kf_counter
                        and self.frontend.is_initialized
                    ):
                        try:
                            pose_file_pth = (
                                f"{self.output}/traj_all/poses_{self.video.counter.value:03d}.npz"
                            )
                            os.makedirs(os.path.dirname(pose_file_pth), exist_ok=True)
                            self.video.save_poses(pose_file_pth)
                            ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                                pose_file_pth,
                                f"{self.output}/traj_all",
                                f"{i:05d}_N_{self.video.counter.value:03d}_kf_traj",
                                stream,
                                None,
                                self.printer,
                            )
                            self.event_writer.add_scalar(
                                "ate_statistics/ate", ate_statistics['rmse'], timestamp
                            )
                            self.event_writer.add_scalar(
                                "ate_statistics/global_scale", global_scale, timestamp
                            )

                        except Exception as e:
                            self.printer.print(e, FontColor.ERROR)

                        self.prev_kf_counter = self.video.counter.value

                if (
                    log_ate_per_kf
                    and not isinstance(stream, RGB_NoPose)
                    and self.video.counter.value > self.prev_kf_counter
                    and self.frontend.is_initialized
                ):
                    try:
                        pose_file_pth = (
                            f"{self.output}/traj_all/poses_{self.video.counter.value:03d}.npz"
                        )
                        os.makedirs(os.path.dirname(pose_file_pth), exist_ok=True)
                        self.video.save_poses(pose_file_pth)
                        ate_statistics, global_scale, _, _ = kf_traj_ate(
                            pose_file_pth,
                            stream,
                            self.printer,
                        )
                        if self.wandb_logger is not None:
                            self.wandb_logger.log(
                                {
                                    "trajectory/kf_ate_rmse": float(ate_statistics["rmse"]),
                                    "trajectory/kf_ate_mean": float(ate_statistics["mean"]),
                                    "trajectory/kf_ate_median": float(ate_statistics["median"]),
                                    "trajectory/kf_global_scale": float(global_scale),
                                    "trajectory/kf_count": int(self.video.counter.value),
                                },
                                step=int(timestamp),
                            )
                    except Exception as e:
                        self.printer.print(e, FontColor.ERROR)
                    self.prev_kf_counter = self.video.counter.value

                if (
                    (starting_count < self.video.counter.value)
                    and self.cfg['mapping']['full_resolution']
                ):
                    if self.motion_filter.uncertainty_aware:
                        img_full = stream.get_color_full_resol(i)
                        self.motion_filter.get_img_feature(timestamp, img_full, suffix='full')
            curr_kf_idx = self.video.counter.value - 1

            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                if self.video.counter.value == self.frontend.warmup:
                    # initialize the second stage of the frontend
                    self.frontend.initialize_second_stage(self.event_writer)
                    ## We just finish the initialization
                    if self.cfg['mapping']['enable']:
                        self.pipe.send(
                            {
                                "is_keyframe": True,
                                "video_idx": curr_kf_idx,
                                "timestamp": timestamp,
                                "just_initialized": True,
                                "end": False,
                            }
                        )
                        self.pipe.recv()
                else:
                    if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                        with timer.section("Online BA"):
                            # run online global BA every {self.ba_freq} keyframes (here, we set as 20 keyframes)
                            self.printer.print(
                                f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",
                                FontColor.TRACKER,
                            )
                            if not self.finish_first_online_ba:
                                self.online_ba.dense_ba(
                                    2,
                                    enable_update_uncer=self.frontend.enable_opt_dyn_mask,
                                    enable_udba=self.frontend.enable_opt_dyn_mask,
                                )
                                self.finish_first_online_ba = True
                            else:
                                self.online_ba.dense_ba(
                                    2,
                                    enable_update_uncer=False,
                                    enable_udba=self.frontend.enable_opt_dyn_mask,
                                )
                        prev_ba_idx = curr_kf_idx
                    # inform the mapper that the estimation of current pose and depth is finished
                    if self.cfg['mapping']['enable']:
                        self.pipe.send(
                            {
                                "is_keyframe": True,
                                "video_idx": curr_kf_idx,
                                "timestamp": timestamp,
                                "just_initialized": False,
                                "end": False,
                            }
                        )
                        self.pipe.recv()

            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()
            self._log_tracking_step(frame_idx=int(timestamp), frame_dt=time.perf_counter() - frame_t0)

        self.pipe.send(
            {
                "is_keyframe": True,
                "video_idx": None,
                "timestamp": None,
                "just_initialized": False,
                "end": True,
            }
        )
