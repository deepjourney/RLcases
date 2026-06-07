import numpy as np
import gymnasium as gym
import cv2, time
import skvideo.io

def _split_step(result):
    """Accept both gymnasium 5-tuple and legacy 4-tuple step returns."""
    if len(result) == 5:
        obs, rew, terminated, truncated, info = result
    else:
        obs, rew, done, info = result
        terminated, truncated = done, False
    return obs, rew, terminated, truncated, info

def _split_reset(result):
    """Accept both gymnasium (obs, info) and legacy obs-only reset returns."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}

class Recorder(gym.Wrapper):
    def __init__(self, env, i, args):
        gym.Wrapper.__init__(self, env=env)
        self.i, self.frewards, self.flengths = i, open(args.rewardsname+str(i),'a'), open(args.lengthsname+str(i),'a')
        self.fexinfos = open(args.rewardsname+'exinfos_'+str(i),'a')
        self.g_step = args.start_step
        self.g_step_plus = args.env_num
        self.timer  = args.timer
        if self.timer: self.step_crt = 0
    def __del__(self):
        if self.timer: print(self.i,'step_crt:',np.round(self.step_crt/60,2),' minutes')
        for file in [self.frewards,self.flengths,self.fexinfos]:
            print('',file=file,flush=True)
            file.close()
    def process_time(self):
        return np.array([time.process_time(),time.perf_counter()])
    def reset(self, **kwargs):
        obs, info = _split_reset(self.env.reset(**kwargs))
        self.reward, self.length, self.last_epreward, self.last_eplength, self.rewlist, self.actlist = 0, 0, -1, -1, [], []
        return obs, info
    def step(self, act):
        if self.timer: self.step_start = self.process_time()
        obs, rew, terminated, truncated, info = _split_step(self.env.step(act))
        done = terminated or truncated
        self.g_step += self.g_step_plus
        self.reward += np.sum(np.asarray(rew))
        self.length += 1
        if done:
            print(int(self.g_step),',',int(self.reward),end='|',file=self.frewards,flush=True)
            print(int(self.g_step),',',int(self.length),end='|',file=self.flengths,flush=True)
            if 'exinfos' in info: print(int(self.g_step),',',info['exinfos'], end='|',file=self.fexinfos,flush=True)
            info = {'last_score':int(self.reward),'last_length':int(self.length),**info}
        if self.timer: self.step_crt += self.process_time()-self.step_start
        return obs, rew, terminated, truncated, info

class Monitor(gym.Wrapper):
    def __init__(self, env, i, args, prefix=''):
        gym.Wrapper.__init__(self, env=env)
        self.gameflag = args.gameflag
        videoname     = args.output_dir+prefix+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str(i)+'.mp4'
        fps, fourcc   = args.fps, cv2.VideoWriter_fourcc(*'mp4v')
        if hasattr(env, 'video_size'): obshape = env.video_size
        else: obshape = self.env.observation_space.shape
        if hasattr(env, 'num_screen'): num_screen = env.num_screen
        else: num_screen = [1,1]
        if len(obshape) == 3: width, height = obshape[0]*args.zoom_in*num_screen[0], obshape[1]*args.zoom_in*num_screen[1]
        else:                 width, height = args.width, args.height
        if self.gameflag == 'atari': self.encoder = ImageEncoder(videoname, (width, height, 3), fps)
        else:                        self.vWriter  = cv2.VideoWriter(videoname, fourcc, fps, (width, height))
        self.debug = args.debug
    def __del__(self):
        if self.gameflag == 'atari': pass
        else:                        self.vWriter.release()
    def render(self, *args, **kwargs):
        frame = self.env.render()
        if self.gameflag == 'atari': self.encoder.capture_frame(frame)
        else:                        self.vWriter.write(frame)
        return frame
    def reset(self, **kwargs):
        obs, info = _split_reset(self.env.reset(**kwargs))
        return obs, info
    def step(self, act):
        obs, rew, terminated, truncated, info = _split_step(self.env.step(act))
        if self.debug: print(obs)
        return obs, rew, terminated, truncated, info

class wrap_deepmind_render(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
    def render(self, *args, **kwargs):
        frame = self.env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        frame = np.expand_dims(frame, -1)
        frame = np.tile(frame, 3)
        return frame
    def reset(self, **kwargs):
        obs, info = _split_reset(self.env.reset(**kwargs))
        return obs, info
    def step(self, act):
        return _split_step(self.env.step(act))

###########################################################################
from stable_baselines3.common.vec_env import VecNormalize as VecNormalize_
class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True
    def train(self):
        self.training = True
    def eval(self):
        self.training = False

###########################################################################
import os, subprocess
class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        h, w, pixfmt = frame_shape
        if pixfmt not in (3, 4):
            raise ValueError("Frame shape {} must have 3 or 4 channels".format(frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec
        self.backend = 'ffmpeg'
        self.start()

    def start(self):
        self.cmdline = (
            self.backend, '-nostats', '-loglevel', 'error', '-y',
            '-r', '%d' % self.frames_per_sec,
            '-f', 'rawvideo',
            '-s:v', '{}x{}'.format(*self.wh),
            '-pix_fmt', 'rgb32' if self.includes_alpha else 'rgb24',
            '-i', '-',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            self.output_path
        )
        if hasattr(os, 'setsid'):
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise ValueError('Wrong type {} for frame'.format(type(frame)))
        if frame.shape != self.frame_shape:
            raise ValueError("Frame shape {} doesn't match expected {}".format(frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise ValueError("Frame dtype {} must be uint8".format(frame.dtype))
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
