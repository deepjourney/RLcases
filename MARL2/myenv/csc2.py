import numpy as np
import random, gym, gym.spaces, json, easydict, time, sys, cv2
from pprint import pprint
from myenv import *

from pysc2.env import sc2_env
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from s2clientprotocol import sc2api_pb2 as sc_pb
class Csc2(gym.Env):
    def __init__(self):
        flags.FLAGS(sys.argv[:1])
        map_name = 'BuildMarines'#'DefeatZerglingsAndBanelings'#'MoveToBeacon'
        agt_name = 'Jet'
        agt_race = 'random'
        bot_race = 'random'
        bot_level= 'very_easy'
        bot_build= 'random'
        map_inst = maps.get(map_name)
        players = []
        players.append(sc2_env.Agent(sc2_env.Race[agt_race], agt_name))
        if map_inst.players >= 2:
            players.append(sc2_env.Bot(sc2_env.Race[bot_race], sc2_env.Difficulty[bot_level], sc2_env.BotBuild[bot_build]))
        sc2env = sc2_env.SC2Env(map_name=map_name,
                                #battle_net_map=False,
                                players=players,
                                agent_interface_format=sc2_env.parse_agent_interface_format(
                                    feature_screen=84, feature_minimap=64, action_space='FEATURES',
                                    rgb_screen=[2048,1152], rgb_minimap=128, #action_space='RGB',#'RAW'
                                    use_feature_units=False, use_raw_units=False),
                                step_mul=8, game_steps_per_episode=None, disable_fog=None, visualize=True)
        self.sc2env = available_actions_printer.AvailableActionsPrinter(sc2env)
        #print('observation_spec')
        #pprint(self.sc2env.observation_spec()[0])
        #print('action_spec')
        #print(self.sc2env.action_spec()[0].types)
        #print(self.sc2env.action_spec()[0].functions)
        self.action_spec = self.sc2env.action_spec()[0]
        self.reset()
        self.observation_space = None#gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,self.map_channel],dtype=np.uint8)
        self.action_space      = None#gym.spaces.Discrete(self.act_agt)
        self.reward_range      = [0,1]
        self.attr = {}
        self.video_size = [2880, 1476, 3]
    def __del__(self):
        pass
        #self.sc2env.save_replay('Jet1')
    def reset(self):
        timesteps = self.sc2env.reset()
        #print(timesteps[0].step_type)
        #print(timesteps[0].reward)
        #print(timesteps[0].discount)
        #pprint(timesteps[0].observation)
        return timesteps[0].observation['feature_screen']

    def step(self, function_ids):
        #print('function_ids',function_ids)
        acts = []
        for function_id in function_ids:
            args = []
            for arg in self.action_spec.functions[function_id].args:
                a = []
                for size in arg.sizes:
                    b = np.random.randint(0, size)
                    a.append(b)
                args.append(a)
            act = actions.FunctionCall(function_id, args)
            acts.append(act)
        #print('acts',acts)
        timesteps = self.sc2env.step(acts)
        return timesteps[0].observation['feature_screen'],timesteps[0].reward,False,{'a_actions':timesteps[0].observation['available_actions']}

    def render(self, mode='rgb_array', close=False):
        self.sc2env._env._renderer_human._obs_queue.join()
        frame = self.sc2env._env._renderer_human._window.copy()#.render(self.sc2env._env._obs[0])
        #frame = np.transpose(pygame.surfarray.pixels3d(frame), axes=(1, 0, 2))
        frame = pygame.surfarray.array3d(frame).swapaxes(0,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) ###cv2 or skvideo
        
        #print(frame.shape)
        #exit()
        #pass
        return frame

    def close(self):
        pass
    def seed(self, seed=None):
        random.seed(seed)
