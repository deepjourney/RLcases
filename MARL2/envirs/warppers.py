import numpy as np
import gym, cv2, time
import skvideo.io
class Recorder(gym.Wrapper):
    def __init__(self, env, i, args):
        gym.Wrapper.__init__(self, env=env)
        self.i, self.frewards, self.flengths = i, open(args.rewardsname+str(i),'a'), open(args.lengthsname+str(i),'a')
        self.fexinfos = open(args.rewardsname+'exinfos_'+str(i),'a')
        #self.frewlist, self.factlist = open(args.rewardsname+'rewlist_'+str(i),'a'), open(args.rewardsname+'actlist_'+str(i),'a')
        self.g_step = args.start_step
        self.g_step_plus = args.env_num
        self.timer  = args.timer
        if self.timer: self.step_crt = 0
    def __del__(self):
        if self.timer: print(self.i,'step_crt:',np.round(self.step_crt/60,2),' minutes')
        for file in [self.frewards,self.flengths,self.fexinfos]:#,self.frewlist,self.factlist]:
            print('',file=file,flush=True)
            file.close()
    def process_time(self):
        return np.array([time.process_time(),time.perf_counter()])
    def reset(self):
        obs = self.env.reset()
        self.reward, self.length, self.last_epreward, self.last_eplength, self.rewlist, self.actlist = 0, 0, -1, -1, [], []
        return obs
    def step(self, act):
        if self.timer: self.step_start = self.process_time()
        obs, rew, done, info = self.env.step(act)
        self.g_step+=self.g_step_plus
        self.reward+=np.sum(np.asarray(rew))
        #print('Recorder',rew,self.reward)
        self.length+=1
        #self.rewlist.append(rew)
        #self.actlist.append(act)
        if done:
            #print('done Recorder',self.reward)
            print(int(self.g_step),',',int(self.reward),end='|',file=self.frewards,flush=True)
            print(int(self.g_step),',',int(self.length),end='|',file=self.flengths,flush=True)
            if 'exinfos' in info: print(int(self.g_step),',',info['exinfos'], end='|',file=self.fexinfos,flush=True)
            #self.last_epreward = int(self.reward)
            #self.last_eplength = int(self.length)
            info = {'last_score':int(self.reward),'last_length':int(self.length),**info}
            #obs = self.reset()
        if self.timer: self.step_crt += self.process_time()-self.step_start
        return obs, rew, done, info
class Monitor(gym.Wrapper):
    def __init__(self, env, i, args, prefix=''):
        gym.Wrapper.__init__(self, env=env)
        self.gameflag= args.gameflag
        videoname    = args.output_dir+prefix+str(args.learnflag)+'_'+str(args.env_seed)+'_'+str(i)+'.mp4'
        fps, fourcc  = args.fps, cv2.VideoWriter_fourcc(*'mp4v')#'M','J','P','G')
        if hasattr(env, 'video_size'): obshape = env.video_size
        else: obshape = self.env.observation_space.shape
        if hasattr(env, 'num_screen'): num_screen = env.num_screen
        else: num_screen = [1,1]
        if len(obshape) == 3: width, height= obshape[0]*args.zoom_in*num_screen[0], obshape[1]*args.zoom_in*num_screen[1]
        else:                 width, height= args.width, args.height
        if self.gameflag=='atari': self.encoder = ImageEncoder(videoname, (width, height, 3), fps)
        #else:                      self.vWriter = skvideo.io.FFmpegWriter(videoname,outputdict={"-r": str(fps)})
        else:                      self.vWriter = cv2.VideoWriter(videoname, fourcc, fps, (width, height)) ###cv2 or skvideo
        self.debug = args.debug
    def __del__(self):
        if self.gameflag=='atari':  pass
        #else:                       self.vWriter.close()
        else:                       self.vWriter.release() ###cv2 or skvideo
    def render(self,mode):
        frame = self.env.render(mode)#mode='rgb_array'
        if self.gameflag=='atari':  self.encoder.capture_frame(frame)
        #else:                       self.vWriter.writeFrame(frame)
        else:                       self.vWriter.write(frame) ###cv2 or skvideo
        return frame
    def reset(self):
        obs = self.env.reset()
        return obs
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        if self.debug: print(obs)
        return obs, rew, done, info
class wrap_deepmind_render(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
    def render(self,mode):
        frame = self.env.render(mode)#mode='rgb_array'
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)#width,height
        frame = np.expand_dims(frame, -1)
        frame = np.tile(frame,3)
        return frame
    def reset(self):
        obs = self.env.reset()
        return obs
    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        return obs, rew, done, info
###########################################################################
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True
    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs-self.ob_rms.mean) / np.sqrt(self.ob_rms.var+self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
###########################################################################
import json,os,subprocess,tempfile,six
import os.path, distutils.spawn, distutils.version
from gym import error, logger
class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise error.InvalidFrame("Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(frame_shape))
        self.wh = (w,h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise error.DependencyNotInstalled("""Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend':self.backend,
            'version':str(subprocess.check_output([self.backend, '-version'],
                                                  stderr=subprocess.STDOUT)),
            'cmdline':self.cmdline
        }

    def start(self):
        self.cmdline = (self.backend,
                     '-nostats',
                     '-loglevel', 'error', # suppress warnings
                     '-y',
                     '-r', '%d' % self.frames_per_sec,

                     # input
                     '-f', 'rawvideo',
                     '-s:v', '{}x{}'.format(*self.wh),
                     '-pix_fmt',('rgb32' if self.includes_alpha else 'rgb24'),
                     '-i', '-', # this used to be /dev/stdin, which is not Windows-friendly

                     # output
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     self.output_path
                     )

        logger.debug('Starting ffmpeg with "%s"', ' '.join(self.cmdline))
        if hasattr(os,'setsid'): #setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise error.InvalidFrame('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise error.InvalidFrame("Your frame has shape {}, but the VideoRecorder is configured for shape {}.".format(frame.shape, self.frame_shape))
        if frame.dtype != np.uint8:
            raise error.InvalidFrame("Your frame has data type {}, but we require uint8 (i.e. RGB values from 0-255).".format(frame.dtype))

        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            logger.error("VideoRecorder encoder exited with status {}".format(ret))
"""class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype)
    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])"""
"""        if self.move==1:
            self.mask = np.roll(self.mask, shift=-self.unit, axis=0)
            self.mask = np.roll(self.mask, shift=-self.unit, axis=1)
        if self.move==2:
            self.mask = np.roll(self.mask, shift=-self.unit, axis=0)
        if self.move==3:
            self.mask = np.roll(self.mask, shift=-self.unit, axis=0)
            self.mask = np.roll(self.mask, shift= self.unit, axis=1)
        if self.move==4:
            self.mask = np.roll(self.mask, shift=-self.unit, axis=1)
        if self.move==5:
            pass
        if self.move==6:
            self.mask = np.roll(self.mask, shift= self.unit, axis=1)
        if self.move==7:
            self.mask = np.roll(self.mask, shift= self.unit, axis=0)
            self.mask = np.roll(self.mask, shift=-self.unit, axis=1)
        if self.move==8:
            self.mask = np.roll(self.mask, shift= self.unit, axis=0)
        if self.move==9:
            self.mask = np.roll(self.mask, shift= self.unit, axis=0)
            self.mask = np.roll(self.mask, shift= self.unit, axis=1)"""