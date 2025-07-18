import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.algorithms.utils.softgnn import Perception_Graph, LinearAssignment
from onpolicy.algorithms.utils.GNN.graph import Topk_Graph


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, use_macro, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.use_macro = use_macro
        self.tpdv = dict(dtype=torch.float32, device=device)
        obs_shape = get_shape_from_obs_space(obs_space)
        if 'Dict' in obs_shape.__class__.__name__:
            self._mixed_obs = True
            if use_macro:
                self.base = Perception_Graph(args.num_agents)
            else:
                self.base = Topk_Graph(args.num_agents, args.hidden_size, args.use_attn, device)
        else:
            self._mixed_obs = False
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

        self.input_size = self.base.output_size

        if (self._use_naive_recurrent_policy or self._use_recurrent_policy) and (not self.use_macro):
            self.rnn = RNNLayer(self.input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            # self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.input_size = self.hidden_size

        self.act = ACTLayer(action_space, self.input_size, self._use_orthogonal, self._gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        # if available_actions is not None:
        #     available_actions = check(available_actions).to(**self.tpdv)
        if available_actions is not None:
            # MultiDiscrete のときは list、Discrete のときは ndarray/Tensor
            if isinstance(available_actions, list):
                available_actions = [check(a).to(**self.tpdv) for a in available_actions]
            else:
                available_actions = check(available_actions).to(**self.tpdv)


        actor_features = self.base(obs)
        # print("actor_features:", actor_features.shape)
        



        if (self._use_naive_recurrent_policy or self._use_recurrent_policy) and (not self.use_macro):
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # print("actor_features.shape:", actor_features.shape)
        # print("self.input_size:", self.input_size)
        # # print("action_out weight shape:", self.act.action_out.fc_mean.weight.shape)        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        # if available_actions is not None:
        #     available_actions = check(available_actions).to(**self.tpdv)
        if available_actions is not None:
            # MultiDiscrete のときは list、Discrete のときは ndarray/Tensor
            if isinstance(available_actions, list):
                available_actions = [check(a).to(**self.tpdv) for a in available_actions]
            else:
                available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if (self._use_naive_recurrent_policy or self._use_recurrent_policy) and (not self.use_macro):
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, use_macro, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.use_macro = use_macro
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        if 'Dict' in cent_obs_shape.__class__.__name__:
            self._mixed_obs = True
            if use_macro:
                self.base = Perception_Graph(args.num_agents)
            else:
                self.base = Topk_Graph(args.num_agents, args.hidden_size, args.use_attn, device)
        else:
            self._mixed_obs = False
            base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
            self.base = base(args, cent_obs_shape)
        
        self.input_size = self.base.output_size

        if (self._use_naive_recurrent_policy or self._use_recurrent_policy) and (not self.use_macro):
            self.rnn = RNNLayer(self.input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            # self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            self.input_size = self.hidden_size
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.input_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.input_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        if self._mixed_obs:
            for key in cent_obs.keys():        
                cent_obs[key] = check(cent_obs[key]).to(**self.tpdv)
        else:
            cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if (self._use_naive_recurrent_policy or self._use_recurrent_policy) and (not self.use_macro):
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        values = self.v_out(critic_features)

        return values, rnn_states
