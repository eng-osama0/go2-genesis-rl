import math
import torch
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def rand(lo, hi, shape, device):
    return (hi - lo) * torch.rand(size=shape, device=device) + lo


class Go2Env:
    """بيئة مشي Unitree Go2: يتعلم الروبوت اتباع أمر سرعة أمامية."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.simulate_action_latency = True
        self.dt = 0.02  # 50Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = dict(reward_cfg["reward_scales"])

        # ---------- المشهد ----------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )
        self.scene.build(n_envs=num_envs)

        # ---------- المفاصل ----------
        self.motors_dof_idx = [
            self.robot.get_joint(n).dof_start for n in env_cfg["dof_names"]
        ]
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # ---------- المكافآت ----------
        self.reward_functions, self.episode_sums = {}, {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(num_envs, device=self.device)

        # ---------- المخازن ----------
        f = dict(device=self.device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((num_envs, 3), **f)
        self.base_ang_vel = torch.zeros((num_envs, 3), **f)
        self.projected_gravity = torch.zeros((num_envs, 3), **f)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], **f).repeat(num_envs, 1)
        self.obs_buf = torch.zeros((num_envs, self.num_obs), **f)
        self.rew_buf = torch.zeros((num_envs,), **f)
        self.reset_buf = torch.ones((num_envs,), device=self.device, dtype=torch.int32)
        self.episode_length_buf = torch.zeros((num_envs,), device=self.device, dtype=torch.int32)
        self.commands = torch.zeros((num_envs, self.num_commands), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f
        )
        self.actions = torch.zeros((num_envs, self.num_actions), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((num_envs, 3), **f)
        self.base_quat = torch.zeros((num_envs, 4), **f)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][n] for n in env_cfg["dof_names"]], **f
        )
        self.extras = {"observations": {}}

    # ---------- أوامر السرعة ----------
    def _resample_commands(self, envs_idx):
        c = self.command_cfg
        n = len(envs_idx)
        if n == 0:
            return
        self.commands[envs_idx, 0] = rand(*c["lin_vel_x_range"], (n,), self.device)
        self.commands[envs_idx, 1] = rand(*c["lin_vel_y_range"], (n,), self.device)
        self.commands[envs_idx, 2] = rand(*c["ang_vel_range"], (n,), self.device)

    # ---------- خطوة المحاكاة ----------
    def step(self, actions):
        clip = self.env_cfg["clip_actions"]
        self.actions = torch.clip(actions, -clip, clip)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat
            ),
            rpy=True,
            degrees=True,
        )
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # إعادة توليد الأوامر
        period = int(self.env_cfg["resampling_time_s"] / self.dt)
        idx = (self.episode_length_buf % period == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(idx)

        # شروط الإنهاء
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = timeout.clone()
        self.reset_buf |= torch.abs(base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.extras["time_outs"] = timeout.float()
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # المكافآت
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # الملاحظات (45 قيمة)
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],               # 3
                self.projected_gravity,                                       # 3
                self.commands * self.commands_scale,                          # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                    # 12
                self.actions,                                                 # 12
            ],
            dim=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ---------- دوال المكافأة ----------
    def _reward_tracking_lin_vel(self):
        err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
