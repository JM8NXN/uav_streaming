import math
import numpy as np
import gymnasium as gym
from itertools import product
from scipy.spatial import distance
import gym
from gym import spaces 


class CommunicationEnv(gym.Env):
    def __init__(
        self,
        args,
        **kwargs
    ):
        """
        Initialize the Communication Environment.
        """
        self.debug = False
        self.test_mode = kwargs.get("test_mode", False)
        self.shared_reward = True
        self.stats_last_n_subcarriers = 0

        self.n_uavs = args.num_uavs
        self.n_ctls = args.controller_num_agents
        self.n_exes = args.num_agents
        self.h_area = 1000
        self.v_area = 1000
        self.z_UAV = 300
        self.m_tiles = 18  # 動画の縦方向のタイル数
        self.n_tiles = 36  # 動画の横方向のタイル数
        self.fov_m_tiles = 10
        self.fov_n_tiles = 10
        self.video_quality_max = 5
        self.episode_limit = 100

        # サブキャリア
        self.n_subcarriers = 64  # サブキャリアの数
        self.n_fixed_subcarriers = 1
        self.n_flexible_subcarriers = self.n_subcarriers - self.n_fixed_subcarriers * self.n_uavs
        self.flexible_subcarriers = []

        # LOS/NLOS時のシャドウイング用パラメータ例
        self.mean_eta_los = 1.6
        self.variance_eta_los = 0 # test
        self.mean_eta_nlos = 23
        self.variance_eta_nlos = 0 # test
        self.a = 9.61
        self.b = 0.158
        self.f_c = 2e9  # Hz
        self.c_0 = 3e8    # m/s
        self.P_UAV = 1e-3  # W
        self.total_bandwidth = 20e6
        self.sub_bandwidth = self.total_bandwidth / self.n_subcarriers
        self.noise_power = (1.38e-23)*290

        self.n_agents = 4 * self.n_uavs - 1 

        self.pilot_positions = np.zeros((self.n_uavs, 3)) 
        self.uav_positions = np.zeros((self.n_uavs, 3))
        self.uav_velocities = np.zeros((self.n_uavs, 3))
        self.uav_destination = np.zeros((self.n_uavs, 3))

        self.previous_fov_bitrate_mean = np.zeros(self.n_uavs) # 1ステップ前のFoV内の平均動画品質
        self.p_v = 0.3  # 垂直視角が固定値となる確率

        #self.bitrate_for_video_quality = [6.66e5, 16.18e5, 24.29e5, 32.01e5, 40.23e5]  # 各品質レベルに対するタイルのビットレート
        self.bitrate_for_video_quality = [1028, 2497, 3748, 4940, 6208]  # 各品質レベルに対するタイルのビットレート，35, 40, 45, 50QP
        self.h_mean = 0
        self.v_mean = -np.pi/180*30
        self.h_var = np.pi/180*50 # 水平方向の分散
        self.v_var = np.pi/180*20 # 垂直方向の分散

        self.fov_matrix_1sigma=[
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

        self.fov_matrix_2sigma = [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]]

        self.n_tiles_1sigma = 226
        self.n_tiles_2sigma = 400


        self.n_relay_actions = self.n_uavs + 1
        self.n_subcarriers_actions = math.ceil(self.n_flexible_subcarriers / self.n_uavs) * 2
        self.n_video_actions = self.video_quality_max

        self.n_relay_agents = self.n_uavs
        self.n_subcarriers_agents = self.n_uavs - 1
        self.n_1sigma_video_agents = self.n_uavs
        self.n_2sigma_video_agents = self.n_uavs

        

        self.weight_utility_reward = 1.0
        self.weight_temporal_jitter_penalty = 0.1
        self.weight_spacial_jitter_penalty = 0.1

        self.reward_scale = 1.0e5
        self.shared_reward = True

        self.n_violate = 0

        self.steps = 0

        obs_dim = self._get_obs_size()
        share_obs_dim = self._get_state_size()
        self.ctl_num_agents = self.n_uavs * 2 - 1
        self.exe_num_agents = self.n_uavs * 2
        self.ctl_num_actions = self.n_video_actions * 2
        self.exe_num_actions = self.n_uavs + self.n_subcarriers_actions
        ctl_space = {
            'agent_pos'    : spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.n_uavs, 3), dtype=np.float32),
            'pilot_pos'    : spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(self.n_uavs, 3), dtype=np.float32),
            'rel_dis'      : spaces.Box(low=0.0, high=np.inf,
                                        shape=(self.n_uavs, self.n_uavs, 1), dtype=np.float32),
        }
        self.ctl_observation_space       = [spaces.Dict(ctl_space)]
        self.ctl_share_observation_space = [spaces.Dict(ctl_space)]
        exe_space = {
            'agent_state' : spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(self.n_uavs, 8), dtype=np.float32),
            'target_goal' : spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(self.n_uavs, 2), dtype=np.float32),
        }
        self.exe_observation_space       = [spaces.Dict(exe_space)]
        self.exe_share_observation_space = [spaces.Dict(exe_space)]

        self.share_observation_space = (
            [spaces.Dict(ctl_space)] * self.ctl_num_agents
            + [spaces.Dict(exe_space)] * self.exe_num_agents
        )

        self.ctl_action_dim = self.n_relay_actions + self.n_subcarriers_actions
        self.exe_action_dim = self.n_video_actions * 2 

        self.ctl_action_space = [
            gym.spaces.Box(low=0.0, high=1.0, shape=(self.ctl_action_dim,), dtype=np.float32)
            for _ in range(self.ctl_num_agents)
        ]
        self.exe_action_space = [
            gym.spaces.Box(low=0.0, high=1.0, shape=(self.exe_action_dim,), dtype=np.float32)
            for _ in range(self.exe_num_agents)
        ]

        self.ctl_reward = 0
        self.exe_reward = 0

        self.agent_relay_flag = [False for _ in range(self.n_uavs)]         # リレーするか否か [[エージェント0がリレーするか],[],…]
        self.relay_uav = [None] * self.n_uavs
        self.n_agent_subcarriers = [None] * self.n_uavs        
        self.video_quality_2sigma = [None] * self.n_uavs
        self.video_quality_1sigma = [None] * self.n_uavs

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, self.agent_obs_dim), dtype=np.float32)
        # self.action_space = gym.spaces.Discrete(self.agent_action_dim)

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def reset(self):
        # エピソードの初期化
        self.steps = 0

        self.n_violate = 0
        self.n_relay_status = 0
        self.over_subcarrier_sum = 0
        self.stats_last_n_subcarriers = 0
        self.status_1sigma_quality = 0
        self.status_2sigma_quality = 0
        self.status_n_sent_base_tiles = 0
        self.status_n_sent_1sigma_tiles = 0
        self.status_n_sent_2sigma_tiles = 0

        x_pilots = np.random.uniform(0, self.h_area, size=self.n_uavs)
        y_pilots = np.random.uniform(0, self.v_area, size=self.n_uavs)

        # pilotはz=0固定
        self.pilot_positions[:, 0] = x_pilots
        self.pilot_positions[:, 1] = y_pilots
        self.pilot_positions[:, 2] = 0.0

        # UAVはz=self.z_UAV固定
        self.uav_positions[:, 0] = x_pilots
        self.uav_positions[:, 1] = y_pilots
        self.uav_positions[:, 2] = self.z_UAV

        # 行き先を生成
        if self.test_mode:
            x_dests = np.random.uniform(0, self.h_area, size=self.n_uavs)
            y_dests = np.random.uniform(0, self.v_area, size=self.n_uavs)
        else:
            x_dests = np.roll(x_pilots, 1)
            y_dests = np.roll(y_pilots, 1)            


        self.uav_destination[:, 0] = x_dests
        self.uav_destination[:, 1] = y_dests
        self.uav_destination[:, 2] = self.z_UAV

        # 速度 = (目的地 - 現在地) / episode_limit
        self.uav_velocities[:, 0] = (x_dests - self.uav_positions[:, 0]) / self.episode_limit
        self.uav_velocities[:, 1] = (y_dests - self.uav_positions[:, 1]) / self.episode_limit
        self.uav_velocities[:, 2] = 0.0

        self.ctl_observation_space = self._get_obs('ctl')
        self.exe_observation_space = self._get_obs('exe')

        self.ctl_share_observation_space = self._get_state('ctl')
        self.exe_share_observation_space = self._get_state('exe')
        self.share_observation_space = self._get_state()

    def step(self, data):
        """
        data: The tuple of (action, mode).
        """
        actions, mode = data

        self.relay_uav = [None] * self.n_uavs
        self.n_agent_subcarriers = [None] * self.n_uavs        
        self.video_quality_2sigma = [None] * self.n_uavs
        self.video_quality_1sigma = [None] * self.n_uavs

        # Decide relay and subcarrier selections
        if mode == 'ctl':
            self.violate_flag = False

            for agent_index in range(self.n_ctls):
                if(agent_index < self.n_relay_agents):
                    self.relay_uav[agent_index] = actions[agent_index]
                else:
                    # one‑hot になっている action ベクトル → argmax で整数 ID に変換
                    sub_idx = int(np.argmax(actions[agent_index]))
                    # 「柔軟サブキャリア ID ＋固定ぶん」をその UAV の本数として保持
                    self.n_agent_subcarriers[agent_index - self.n_relay_agents] = (
                        sub_idx - self.n_relay_actions + self.n_fixed_subcarriers
                    )

            n_rest_subcarriers = self.n_flexible_subcarriers
            for uav_index in range(self.n_uavs - 1):
                n_rest_subcarriers -= self.n_agent_subcarriers[uav_index] - self.n_fixed_subcarriers
            self.n_agent_subcarriers[self.n_uavs - 1] = n_rest_subcarriers + self.n_fixed_subcarriers

            # over the limitation of the number of subcarriers
            if self.n_agent_subcarriers[-1] < self.n_fixed_subcarriers:
                self.violate_flag = True
                self.n_violate += 1
                over_subcarrier = self.n_fixed_subcarriers - self.n_agent_subcarriers[-1]
                self.over_subcarrier_sum += over_subcarrier
                # video stop
                for uav_index in range(self.n_uavs):
                    self.previous_fov_bitrate_mean[uav_index] = 0
                # Update the position of the UAVs
                self.uav_positions += self.uav_velocities        
                self.ctl_reward = -over_subcarrier / self.n_subcarriers_actions / self.n_uavs * 2
            else:
                self.stats_last_n_subcarriers += int(self.n_agent_subcarriers[-1])
            self.ctl_reward = 0
        elif mode == 'exe':
            for agent_index in range(self.n_exes):
                if(agent_index < self.n_1sigma_video_agents):
                    self.video_quality_1sigma[agent_index] = actions[agent_index]
                else:
                    self.video_quality_2sigma[agent_index - self.n_1sigma_video_agents] = actions[agent_index] - self.n_video_actions

            # 各エージェントの伝送レート計算
            # それぞれのエージェントがどのサブキャリアを使用するか，そして干渉についての処理
            
            self.exe_reward = 0

            for uav_index in range(self.n_uavs):
                self.status_1sigma_quality += self.video_quality_1sigma[uav_index]
                self.status_2sigma_quality += self.video_quality_2sigma[uav_index]

            # リレーするかどうかのフラグ
            for uav_index, self.relay_uav_agent in enumerate(self.relay_uav):      # self.relay_uav_agentはint型
                if self.relay_uav_agent != 0:
                    self.agent_relay_flag[uav_index] = True
                    self.n_relay_status += 1

            transmission_rate_a2g = [None] * self.n_uavs              # UAV-pilot間通信での伝送レート
            transmission_rate_a2a = [None] * self.n_uavs              # UAV(リレー元)-UAV(リレー先)間通信での伝送レート
            total_transmission_rate = [None] * self.n_uavs

            for uav_index in range(self.n_uavs):

                # 一つのサブキャリアで得られる伝送レートを計算する。サブキャリアを区別しないものとする。
                if self.agent_relay_flag[uav_index] == False:
                    SNR = self.calculate_snr(self.uav_positions[uav_index], self.pilot_positions[uav_index], "uav-pilot")
                    transmission_rate_a2g[uav_index] = self.sub_bandwidth * math.log2(1 + SNR)
                elif self.agent_relay_flag[uav_index] == True:
                    SNR = self.calculate_snr(self.uav_positions[self.relay_uav[uav_index]-1], self.pilot_positions[uav_index], "uav-pilot")
                    transmission_rate_a2g[uav_index] = self.sub_bandwidth * math.log2(1 + SNR)
                    SNR = self.calculate_snr(self.uav_positions[uav_index], self.uav_positions[self.relay_uav[uav_index]-1], "uav-uav")
                    transmission_rate_a2a[uav_index] = self.sub_bandwidth * math.log2(1 + SNR)
                
                # エージェントの総伝送レートの計算．
                if self.agent_relay_flag[uav_index] == False:
                    total_transmission_rate[uav_index] = transmission_rate_a2g[uav_index] * self.n_agent_subcarriers[uav_index]
                #リレーする場合のA2GとA2Aのサブキャリア配分の最適化．サブキャリアで区別をつけないのでgreedyで最適になる
                elif self.agent_relay_flag[uav_index] == True:
                    total_transmission_rate_a2a = 0
                    total_transmission_rate_a2g = 0
                    # Greedyでサブキャリアを割り当てる
                    for _ in range(self.n_agent_subcarriers[uav_index]):
                        if total_transmission_rate_a2a > total_transmission_rate_a2g:
                            total_transmission_rate_a2g += transmission_rate_a2g[uav_index]
                        else:
                            total_transmission_rate_a2a += transmission_rate_a2a[uav_index]
                    total_transmission_rate[uav_index] = min(total_transmission_rate_a2a, total_transmission_rate_a2g)
                for uav_index in range(self.n_uavs):
                    self.ctl_reward += total_transmission_rate[uav_index]

                # ベースレイヤの送信
                if total_transmission_rate[uav_index] >= self.bitrate_for_video_quality[0] * self.n_tiles * self.m_tiles:
                    total_transmission_rate[uav_index] -= self.bitrate_for_video_quality[0] * self.n_tiles * self.m_tiles
                    self.status_n_sent_base_tiles += 1
                else:
                    self.exe_reward -= 1.0 / self.n_uavs                
                    continue
                sent_tiles = np.ones((self.m_tiles, self.n_tiles), dtype=int)

                # 1-sigma, 2-sigmaの領域で送信可能かの確認
                bitrate_difference_1sigma = self.bitrate_for_video_quality[self.video_quality_1sigma[uav_index]] - self.bitrate_for_video_quality[0]
                bitrate_difference_2sigma = self.bitrate_for_video_quality[self.video_quality_2sigma[uav_index]] - self.bitrate_for_video_quality[0]

                send_able_2sigma_flag = False

                # 送信されるタイルの品質の計算
                if total_transmission_rate[uav_index] >= bitrate_difference_1sigma * self.n_tiles_1sigma: # 1sigma領域のタイルの全てが変換可能か
                    total_transmission_rate[uav_index] -= bitrate_difference_1sigma * self.n_tiles_1sigma
                    self.status_n_sent_1sigma_tiles += 1
                    if total_transmission_rate[uav_index] >= bitrate_difference_2sigma * self.n_tiles_2sigma:   # 2sigma領域のタイルの全てが変換可能か
                        send_able_2sigma_flag = True
                        total_transmission_rate[uav_index] -= bitrate_difference_2sigma * self.n_tiles_2sigma
                        self.status_n_sent_2sigma_tiles += 1
                    for m_index, n_index in product(range(self.m_tiles), range(self.n_tiles)):
                        # 1-sigma領域
                        if self.fov_matrix_1sigma[m_index][n_index] == 1:
                            sent_tiles[m_index][n_index] = self.video_quality_1sigma[uav_index] + 1
                        # 2-sigma領域
                        elif self.fov_matrix_2sigma[m_index][n_index] == 1 and send_able_2sigma_flag:
                            sent_tiles[m_index][n_index] = self.video_quality_2sigma[uav_index] + 1

                # FoV行列の生成．各ステップで(0, 0)としても等価
                fov_tiles_matrix = self.generate_fov_matrix(0, 0)
                
                # FoV内のタイルが送信されたかの確認
                tiles_sent_completion_flag = True
                for fov_tile_item in fov_tiles_matrix:
                    # 送られるべきタイルが送られていない場合
                    if sent_tiles[fov_tile_item[0]][fov_tile_item[1]] == 0:
                        tiles_sent_completion_flag=False
                        break

            spacial_jitter = 0

            current_bitrate_mean = None

            # When all tiles are sent            
            if tiles_sent_completion_flag == True:
                # calculate total bitrate of the tiles in FoV
                fov_total_bitrate = 0
                for tile_index in range(self.fov_m_tiles * self.fov_n_tiles):
                    video_quality = sent_tiles[fov_tiles_matrix[tile_index][0]][fov_tiles_matrix[tile_index][1]] - 1
                    fov_total_bitrate += self.bitrate_for_video_quality[video_quality]
                current_bitrate_mean = fov_total_bitrate / (self.fov_m_tiles * self.fov_n_tiles)
                self.exe_reward += current_bitrate_mean * self.weight_utility_reward

                # calculate temporal jitter penalty
                if self.previous_fov_bitrate_mean[uav_index] != 0:
                    self.exe_reward -= abs(current_bitrate_mean - self.previous_fov_bitrate_mean[uav_index]) * self.weight_temporal_jitter_penalty

                # calculate spacial jitter penalty
                squared_difference = 0
                for fov_tile_item in fov_tiles_matrix:
                    video_quality = sent_tiles[fov_tile_item[0]][fov_tile_item[1]] - 1
                    squared_difference += (self.bitrate_for_video_quality[video_quality] - current_bitrate_mean) ** 2 
                spacial_jitter = math.sqrt(squared_difference / (self.fov_m_tiles * self.fov_n_tiles) )              
                self.exe_reward -= spacial_jitter * self.weight_spacial_jitter_penalty
            else :
                self.previous_fov_bitrate_mean[uav_index] = 0
                current_bitrate_mean = 0
                self.exe_reward -= 1.0 / self.n_uavs
                

            self.previous_fov_bitrate_mean[uav_index] = current_bitrate_mean

        # 正規化
        self.exe_reward /= self.reward_scale

        # UAVの位置の更新
        self.uav_positions += self.uav_velocities
        
        
        # エピソード終了条件
        terminated = self.steps >= self.episode_limit
        info = {}


        obs_n, reward_n, done_n, info_n = self.get_data(mode)
        return obs_n, reward_n, np.array(done_n), info
    
    def calculate_snr(self, send_position, receive_position, communication_type):
        d_ij = distance.euclidean(send_position, receive_position)

        if d_ij != 0:
            elevation_angle = np.degrees(np.arcsin(abs(receive_position[2] - send_position[2])/ d_ij))
        else:
            elevation_angle = 90

        if communication_type == "uav-uav":
            # UAV-UAVの場合: 常にLoS
            los_probability = 1.0
        elif communication_type == "uav-pilot":
            # UAV-pilotの場合: LoS確率を計算
            los_probability = 1 / (1 + self.a * np.exp(-self.b * (elevation_angle - self.a)))
        else:
            raise ValueError("communication_type is 'uav-uav' or 'uav-pilot' only")

        # test
        # los_probability = 0

        # メインリンクLoS判定
        if communication_type == "uav-uav":
            los_main = True
        else:
            los_main = (np.random.rand() < los_probability)

        # メインリンクパスロス
        if los_main:
            eta_main = np.random.normal(self.mean_eta_los, np.sqrt(self.variance_eta_los))
        else:
            eta_main = np.random.normal(self.mean_eta_nlos, np.sqrt(self.variance_eta_nlos))

        path_loss_uav = 20 * np.log10(d_ij) + 20 * np.log10(self.f_c) + 20 * np.log10(4 * np.pi / self.c_0) + eta_main
         # path_loss_uav = 32.4 + 10 * (A_parameter * send_position[2] ** B_parameter) np.log10(d_ij) + 20 * np.log10(self.f_c) + 20 * np.log10(4 * np.pi / self.c_0) + eta_main

        noise_power = self.noise_power * self.sub_bandwidth
        signal_power = self.P_UAV * 10**(-path_loss_uav / 10)

        snr = signal_power / noise_power
        #print(f"signal power: {signal_power}\nnoise power: {self.noise_power}\ntotal interference power: {total_interference_power}\nSINR: {snr}\n")
        return snr

    def determine_fov_tiles(self, degree_h, degree_v):
        fov_matrix = []
        m_tile_center = int(degree_v * self.m_tiles + (self.m_tiles-self.fov_m_tiles)//2 )
        n_tile_center = int(degree_h * self.n_tiles + (self.n_tiles-self.fov_n_tiles)//2 )

        for m in range(self.fov_m_tiles):
            for n in range(self.fov_n_tiles):
                check_m = m + m_tile_center - self.m_tiles // 2
                check_n = n + n_tile_center - self.n_tiles // 2

                # 横方向のラッピング
                if check_n < 0 or check_n >= self.n_tiles:
                    check_n = (check_n + self.n_tiles) % self.n_tiles

                # 縦方向のラッピング
                if check_m < 0 or check_m >= self.m_tiles:
                    check_m = ((2 * self.m_tiles - check_m -1) % self.m_tiles + self.m_tiles) % self.m_tiles
                    check_n = ((check_n + self.n_tiles//2) % self.n_tiles + self.n_tiles) % self.n_tiles

                fov_matrix.append([int(check_m), int(check_n)])
        return fov_matrix

    def generate_fov_matrix(self, h_direction, v_direction):
        h_std = np.sqrt(self.h_var)
        phi_p = np.random.normal(h_direction, h_std)
        
        if np.random.rand() < self.p_v:
            theta_p = v_direction
        else:
            v_std = np.sqrt(self.v_var)
            theta_p = np.random.normal(v_direction - self.v_mean, v_std)

        degree_h = (np.pi + phi_p) / (2 * np.pi)
        degree_v = (np.pi / 2 - theta_p) / np.pi
        
        fov_matrix = self.determine_fov_tiles(degree_h, degree_v)
        return fov_matrix

    def get_data(self, mode):
        obs_n = self._get_obs(mode)
        reward_n = self._get_reward(mode)
        done_n = self._get_done()
        info_n = []

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * (self.ctl_num_agents + self.exe_num_agents)

        return obs_n, reward_n, done_n, info_n    

    def _get_obs(self, mode):
        if mode == 'ctl':
            # ① 各エージェント個別の dict を作る
            agents_obs = [self._get_obs_agent(i, mode) for i in range(self.n_ctls)]

            # ② object 型の空配列に詰める   shape = (n_ctls,)
            obs_array = np.empty(self.n_ctls, dtype=object)
            for idx, d in enumerate(agents_obs):
                obs_array[idx] = d
            return obs_array            # ← Runner では obs[e, a] で dict 取得可

        elif mode == 'exe':
            agents_obs = [self._get_obs_agent(i, mode) for i in range(self.n_exes)]
            obs_array = np.empty(self.n_exes, dtype=object)
            for idx, d in enumerate(agents_obs):
                obs_array[idx] = d
            return obs_array

    def _get_obs_agent(self, agent_number, mode=None):
        uav_idx = agent_number % self.n_uavs
        # own 3D position
        own_uav_pos = self.uav_positions[uav_idx].astype(np.float32)           # (3,)
        own_plt_pos = self.pilot_positions[uav_idx].astype(np.float32)

        if mode == 'ctl':
            # other UAVs positions
            other_uav_pos = np.vstack([
                self.uav_positions[i]
                for i in range(self.n_uavs)
                if i != uav_idx
            ]).astype(np.float32)  

            other_plt_pos = np.vstack([
                self.pilot_positions[i]
                for i in range(self.n_uavs)
                if i != uav_idx
            ]).astype(np.float32)  

            # 全 UAVとpilot の位置
            all_uav_pos = np.insert(other_uav_pos, 0, own_uav_pos, axis=0)         # shape = (N, 3)
            all_plt_pos = np.insert(other_plt_pos, 0, own_plt_pos, axis=0) 
            all_plt_pos_xy = all_plt_pos[:, :2]


            # 全対全の差分ベクトルを作成
            diff = all_uav_pos[:, None, :] - all_uav_pos[None, :, :]             # shape = (N, N, 3)

            # 各差分ベクトルのノルムを計算して、チャネル次元を追加
            rel_dis = np.linalg.norm(diff, axis=-1, keepdims=True)

            return {
                'agent_pos'     : all_uav_pos,      # ノード特徴 (optional)
                'pilot_pos'     : all_plt_pos,        # 自分のパイロット位置 (3,)
                'rel_dis'       : rel_dis           # 自分−他 UAV 距離 (N, N, 1)
            }
        elif mode == 'exe':
            relay_flag = self.agent_relay_flag[uav_idx]
            if relay_flag == False:
                relay_pos_xy = np.zeros(2)
            else:
                relay_uav = self.relay_uav[uav_idx]
                relay_pos = self.uav_positions[relay_uav]
                relay_pos_xy = relay_pos[:2]
            n_sbcrs = np.array([self.n_agent_subcarriers[uav_idx]], dtype=np.float32)

            # own previous FoV bitrate mean
            video_quality = np.array([self.previous_fov_bitrate_mean[uav_idx]],
                                    dtype=np.float32)                        # (1,)

            agent_state = np.concatenate([
                own_uav_pos,             # (3,)
                np.array([relay_flag], dtype=np.float32),       # (1,)
                relay_pos_xy,            # (3,)
                n_sbcrs,
                video_quality
            ])
            return {
                'agent_state'  : agent_state,      # ノード特徴 (optional)
                'target_goal'  : own_plt_pos[:2],          # 自分の UAV 位置 (3,)
            }

    def _get_done(self):
        return self.steps >= self.episode_limit

    def _get_reward(self, mode):
        if mode == 'ctl':
            return self.ctl_reward
        if mode == 'exe':
            return self.exe_reward


    def _get_state(self, mode='all'):
        # 全UAVの位置、動画品質、および全地上パイロットの位置を状態として返す
        state = []

        for uav_index in range(self.n_uavs):
            uav_position = self.uav_positions[uav_index]
            previous_video_quality_mean = self.previous_fov_bitrate_mean[uav_index]
            pilot_position = self.pilot_positions[uav_index]


            state.extend(uav_position/100.0)                 # 位置 (3)
            state.append(previous_video_quality_mean/1.0e5)   # video_quality (1)
            state.extend(pilot_position/100.0)                     # パイロットの位置

        state_array = np.array(state, dtype=np.float32) 

        if mode == 'ctl':
            share_obs_list = [state_array.copy() for _ in range(self.n_uavs * 2 - 1)]
        elif mode == 'exe':
            share_obs_list = [state_array.copy() for _ in range(self.n_uavs * 2)]
        else:
            share_obs_list = [state_array.copy() for _ in range(self.n_uavs * 4 - 1)]


        return np.array(share_obs_list).flatten()

    def _get_obs_size(self):
        # obsのサイズ = 自UAV位置 (3) + 動画品質(1) + 自pilot位置(3) + 他UAV位置(3) * (n_uavs - 1)
        return 7 + 3 * (self.n_uavs - 1)

    def _get_state_size(self):
        # 状態のサイズを返す（各エンティティの特徴量のサイズ * エンティティの数）
        feature_size = 10
        return feature_size * self.n_uavs

    def get_avail_agent_actions(self, agent_index, mode):
        """Returns the available actions for agent_index, including relay, video, and subcarrier actions."""

        video_actions = np.zeros(self.n_video_actions * 2)
            
        if mode == 'ctl':
            relay_actions = np.zeros(self.n_uavs)
            subcarriers_actions = np.zeros(self.n_subcarriers_actions)
            if agent_index <= self.n_uavs:
                relay_actions = np.ones(self.n_uavs)
            else:
                subcarriers_actions = np.ones(self.n_subcarriers_actions)
            return  np.concatenate([np.array(action).flatten() for action in [relay_actions, subcarriers_actions]])
        elif mode == 'exe':
            video_actions1 = np.zeros(self.n_video_actions)
            video_actions2 = np.zeros(self.n_video_actions)
            if agent_index <= self.n_uavs:
                video_actions1 = np.ones(self.n_uavs)
            else:
                video_actions2 = np.ones(self.n_uavs)
            return  np.concatenate([np.array(action).flatten() for action in [video_actions1, video_actions2]])

    def get_avail_actions(self, mode):
        avail_actions = []
        for i in range(self.n_agents):
            concatenated_actions  = self.get_avail_agent_actions(i, mode)
            avail_actions.append((concatenated_actions))
        return avail_actions

    def get_total_actions(self):        
        return self.n_relay_actions + self.n_subcarriers_actions + self.n_video_actions * 2
        
    def get_stats(self):
        return {
            "violate_count": self.n_violate,
            "relay_count": self.n_relay_status / self.episode_limit,
            "over_subcarriers" : self.over_subcarrier_sum / self.episode_limit,
            "quality_1sigma" : self.status_1sigma_quality / self.episode_limit / self.n_uavs,
            "quality_2sigma" : self.status_2sigma_quality / self.episode_limit / self.n_uavs,
            "base_sent_completion_count" : self.status_n_sent_base_tiles / self.episode_limit,
            "1sigma_sent_completion_count" : self.status_n_sent_1sigma_tiles / self.episode_limit,
            "2sigma_sent_completion_count" : self.status_n_sent_2sigma_tiles / self.episode_limit,
            "testcheck" : self.test_mode ,
        }

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_env_info(self):
        env_info = {
            "state_shape": self._get_state_size(),
            "obs_shape": self._get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_relay_actions": self.n_relay_actions,
            "n_subcarriers_actions": self.n_subcarriers_actions,
            "n_video_actions": self.n_video_actions,
            "n_agents": self.n_agents,
            "n_relay_agents": self.n_relay_agents,
            "n_subcarrier_agents": self.n_subcarriers_agents,
            "n_1sigma_video_agents": self.n_1sigma_video_agents,
            "n_2sigma_video_agents": self.n_2sigma_video_agents,
            "n_entities": self.n_uavs,
            "episode_limit": self.episode_limit,
            # "n_spare_subcarriers": self.n_spare_subcarriers
        }
        return env_info
    
    def to_one_hot(vec: np.ndarray, num_classes: int) -> np.ndarray:
        """
        vec: 連続値ベクトル（例: shape=(K,)）
        num_classes: ワンホットにしたい次元数（通常は vec.shape[0] と同じ）
        return: shape=(num_classes,) のワンホットベクトル（int型）
        """
        one_hot = np.zeros(num_classes, dtype=np.int32)
        # 最大値のインデックスを選択
        idx = int(np.argmax(vec))
        one_hot[idx] = 1
        return one_hot

        
