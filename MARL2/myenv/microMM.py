import numpy as np
import random, gym, gym.spaces, json, easydict, time, cv2
from collections import deque
from itertools import cycle
import matplotlib
import matplotlib.cm as cm
from myenv._render import *
class MicroMM(gym.Env):
    def __init__(self):
        self.starttime = time.time()
        pygame.init()
        pygame.mixer.quit()
        super().__init__()
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = easydict.EasyDict(json.load(envinfo_file))
        args = easydict.EasyDict()
        args.render         = envinfo_args_dict.render
        args.times          = envinfo_args_dict.zoom_in
        self.rnd_seed       = envinfo_args_dict.rnd_seed
        learnflag           = envinfo_args_dict.learnflag.split('_')
        self.learnflag      = [  int(para) for para in learnflag[0].split(',')]
        switches            = envinfo_args_dict.switches.split('_')
        self.switches       = [float(para) for para in switches[0].split(',')]
        envparas            = envinfo_args_dict.envparas.split('_')
        args.map_size       = [  int(para) for para in envparas[0].split(',')]
        args.end_cond       = [float(para) for para in envparas[1].split(',')]
        args.rew_rng        = [float(para) for para in envparas[2].split(',')]
        self.map_width_small,self.map_height_small,self.map_channel,self.map_grid = args.map_size[0], args.map_size[1], args.map_size[2], args.map_size[3]
        self.map_width, self.map_height = (self.map_width_small-1)*self.map_grid, (self.map_height_small-1)*self.map_grid
        self.end_epsd, self.end_turn = args.end_cond[0], args.end_cond[1]
        npcparas            = envinfo_args_dict.npcparas.split('_')
        args.npc_paras      = [[(para) for para in npcpara.split(',')] for npcpara in npcparas]
        self.min_npc = [int(para[0]) for para in args.npc_paras]
        self.max_npc = [int(para[1]) for para in args.npc_paras]
        self.act_npc = [int(para[2]) for para in args.npc_paras]
        self.fhp_npc = [int(para[3]) for para in args.npc_paras]
        self.atk_npc = [int(para[4]) for para in args.npc_paras]
        self.rng_npc = [int(para[5]) for para in args.npc_paras]
        self.mov_npc = [int(para[6]) for para in args.npc_paras]
        self.num_npc = self.max_npc
        agtparas            = envinfo_args_dict.agtparas.split('_')
        args.agt_paras      = [[(para) for para in agtpara.split(',')] for agtpara in agtparas]
        self.min_agt = [int(para[0]) for para in args.agt_paras]
        self.max_agt = [int(para[1]) for para in args.agt_paras]
        self.act_agt = [int(para[2]) for para in args.agt_paras]
        self.fhp_agt = [int(para[3]) for para in args.agt_paras]
        self.atk_agt = [int(para[4]) for para in args.agt_paras]
        self.rng_agt = [int(para[5]) for para in args.agt_paras]
        self.mov_agt = [int(para[6]) for para in args.agt_paras]
        self.num_agt = self.max_agt
        self.clock = pygame.time.Clock()
        self.envs_list = pygame.sprite.Group()
        self.npcs_list = []
        for i,num_npc in enumerate(self.num_npc):
            self.npcs_list.append(pygame.sprite.Group())
        self.agts_list = []
        for i,num_agt in enumerate(self.num_agt):
            self.agts_list.append(pygame.sprite.Group())
        self.envs_active_list = pygame.sprite.Group()
        self.npcs_active_list = []
        for i,num_npc in enumerate(self.num_npc):
            self.npcs_active_list.append(pygame.sprite.Group())
        self.agts_active_list = []
        for i,num_agt in enumerate(self.num_agt):
            self.agts_active_list.append(pygame.sprite.Group())
        self.screen = pygame.Surface([self.map_width, self.map_height])
        self.lists_visible= []
        self.lists_visible.append(self.envs_active_list)
        for i,num_npc in enumerate(self.num_npc):
            self.lists_visible.append(self.npcs_active_list[i])
        for i,num_agt in enumerate(self.num_agt):
            self.lists_visible.append(self.agts_active_list[i])
        self.screen_small = []
        self.npcarray = np.zeros((self.map_width_small,self.map_height_small,len(self.num_npc)),dtype=np.int8) # np.sum default is int64 not int8
        self.agtarray = np.zeros((self.map_width_small,self.map_height_small,len(self.num_agt)),dtype=np.int8)
        self.crtarray = np.zeros((self.map_width_small,self.map_height_small,1),dtype=np.int8) # int8 is from -128 to 127, good input data +/- balance for 255 colors
        self.layernum = self.npcarray.shape[-1]+self.agtarray.shape[-1]+self.crtarray.shape[-1]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        self.mapper, self.scale = cm.ScalarMappable(norm=norm, cmap=cm.nipy_spectral), 0.1
        #gist_ncar,bwr,nipy_spectral,gnuplot,gnuplot2,gist_earth,gist_stern,inferno,jet,hsv,cm.Greys_r)
        self.model_nums = ['None']*(len(self.num_agt)+len(self.num_npc))
        self.reset()
        if args.render: self.setsize(args.times,0)
        #print(':',time.time()-self.starttime)
        #self.starttime = time.time()
        self.observation_space = None#gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,self.map_channel],dtype=np.uint8)
        self.action_space      = None#gym.spaces.Discrete(self.act_agt)
        self.reward_range      = args.rew_rng ### should be no repeat element list?
        self.attr = {}
        self.attr['unit_num_alls'] = self.num_agt+self.num_npc
        self.attr['unit_num_agts'] = self.num_agt
        self.attr['unit_num_npcs'] = self.num_npc
        self.attr['obs_spaces'] = [gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,self.layernum],dtype=np.uint8)
                                   for i,num_agt in enumerate(self.attr['unit_num_agts'])] \
                                 +[gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,self.layernum],dtype=np.uint8)
                                   for i,num_npc in enumerate(self.attr['unit_num_npcs'])]
        self.attr['act_spaces'] = [gym.spaces.Discrete(self.act_agt[i])
                                   for i,num_agt in enumerate(self.attr['unit_num_agts'])] \
                                 +[gym.spaces.Discrete(self.act_npc[i])
                                   for i,num_agt in enumerate(self.attr['unit_num_npcs'])]
    def __del__(self):
        pygame.quit()
    def setsize(self,times,timelag):
        self.real_screen_width, self.real_screen_height, self.times_small, self.timelag = self.map_width_small*times, self.map_height_small*times, times, timelag
        self.times = self.map_width_small*times//self.map_width
        self.video_size, self.num_screen = [self.map_width_small,self.map_height_small,3], [max(sum(self.num_agt),sum(self.num_npc)),3]
        self.real_screen = pygame.display.set_mode([self.real_screen_width*self.num_screen[0], self.real_screen_height*self.num_screen[1]]) ##,HWSURFACE|DOUBLEBUF|RESIZABLE
    def reset(self):
        self.atklines = []
        self.info, self.done, self.turn, self.epsd = {}, False, 0, 0
        self.gain, self.loss, self.miss = 0, 0, 0
        self.map_generate()
        self.get_observation()
        return self.observation
    def map_generate(self):
        self.envs_list.empty()
        for i,num_npc in enumerate(self.num_npc):
            self.npcs_list[i].empty()
        for i,num_agt in enumerate(self.num_agt):
            self.agts_list[i].empty()
        self.envs_active_list.empty()
        for i,num_npc in enumerate(self.num_npc):
            self.npcs_active_list[i].empty()
        for i,num_agt in enumerate(self.num_agt):
            self.agts_active_list[i].empty()
        colorbar = cycle(['red1','green1','blue', 'yellow1','magenta', 'cyan']) #no aqua in new version
        colorbar1= cycle(['red1','green1','blue', 'yellow1','magenta', 'cyan', 'rosybrown1','darkseagreen1'])#brown,palegreen1
        colorbar4= cycle(['red4','green4','blue4','yellow4','magenta4','cyan4','rosybrown4','darkseagreen4'])
        num_npc = sum(self.num_npc)#random.randrange(self.min_num_npc,self.max_num_npc+1)
        npcplace = random.sample(range(1,self.map_width-1), num_npc)
        for i,num_npc in enumerate(self.num_npc):
            for j in range(num_npc):
                k = sum(self.num_npc[:i])+j#k formula
                color = pygame.color.THECOLORS[next(colorbar1)]
                npc = LPVSprite(ispecies=i,inum=k,color=color,lenx=1,leny=1,posx=npcplace[k],posy=0,hp_max=self.fhp_npc[i],hp_crt=self.fhp_npc[i],atk=self.atk_npc[i],rng=self.rng_npc[i])
                self.npcs_list[i].add(npc)
                if npc.hp_crt>0: self.npcs_active_list[i].add(npc)
        agtplace = random.sample(range(self.map_width), sum(self.num_agt))
        agtplacy = self.map_height-1
        agtzeron = random.randrange(0,sum(self.max_agt)-sum(self.min_agt)+1)###
        agtzeros = random.sample(range(0,sum(self.num_agt)),agtzeron)
        self.speciesloss=[]
        for i,num_agt in enumerate(self.num_agt):
            agthpcrt = []
            for j in range(num_agt):
                k = sum(self.num_agt[:i])+j#k formula
                if k in agtzeros: hpcrt = 0
                else:             hpcrt = self.fhp_agt[i]
                agthpcrt.append(hpcrt)
                color = pygame.color.THECOLORS[next(colorbar4)]
                agt = LPVSprite(ispecies=i,inum=k,color=color,lenx=1,leny=1,posx=agtplace[k],posy=agtplacy,hp_max=self.fhp_agt[i],hp_crt=hpcrt,atk=self.atk_agt[i],rng=self.rng_agt[i])
                self.agts_list[i].add(agt)
                if agt.hp_crt>0: self.agts_active_list[i].add(agt)
            self.speciesloss.append(sum(agthpcrt)==0)
        self.atkorder = random.sample(range(0,sum(self.num_agt)),sum(self.num_agt))
        self.obs_prev, self.repeat = deque(maxlen=10), False
    def square_dist(self,x,y,l):
        X, Y  = x//l,     y//l
        Xl,Yl = x-x//l*l, y-y//l*l
        Xu,Yu = l-Xl,     l-Yl
        return X,Y,Xl,Xu,Yl,Yu,l*l
    def get_observation(self):
        self.screen_small.clear()
        max_units = sum(self.num_npc)
        onehots = np.identity(len(self.num_npc))*255
        self.npcarray.fill(0)
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                X,Y,Xl,Xu,Yl,Yu,Tt = self.square_dist(npc.rect.centerx,npc.rect.centery,self.map_grid)
                self.npcarray[X][Y]     += (Xu*Yu/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[i]/max_units).astype(int)
                self.npcarray[X+1][Y]   += (Xl*Yu/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[i]/max_units).astype(int)
                self.npcarray[X][Y+1]   += (Xu*Yl/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[i]/max_units).astype(int)
                self.npcarray[X+1][Y+1] += (Xl*Yl/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[i]/max_units).astype(int)
        max_units = sum(self.num_agt)
        onehots = np.identity(len(self.num_agt))*255
        self.agtarray.fill(0)
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                X,Y,Xl,Xu,Yl,Yu,Tt = self.square_dist(agt.rect.centerx,agt.rect.centery,self.map_grid)
                self.agtarray[X][Y]     += (Xu*Yu/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[i]/max_units).astype(int)
                self.agtarray[X+1][Y]   += (Xl*Yu/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[i]/max_units).astype(int)
                self.agtarray[X][Y+1]   += (Xu*Yl/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[i]/max_units).astype(int)
                self.agtarray[X+1][Y+1] += (Xl*Yl/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[i]/max_units).astype(int)
        allarray = np.concatenate([self.npcarray,self.agtarray],axis=-1)
        if any((allarray==array).all() for array in self.obs_prev): self.repeat = True
        else: self.obs_prev.append(allarray)
        self.observation = []
        max_units = 1
        onehots = np.identity(1)*255
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                self.crtarray.fill(0)
                X,Y,Xl,Xu,Yl,Yu,Tt = self.square_dist(agt.rect.centerx,agt.rect.centery,self.map_grid)
                self.crtarray[X][Y]     += (Xu*Yu/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X+1][Y]   += (Xl*Yu/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X][Y+1]   += (Xu*Yl/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X+1][Y+1] += (Xl*Yl/Tt*max(0,(agt.hp_crt/agt.hp_max))*onehots[0]/max_units).astype(int)
                array = np.concatenate([allarray,self.crtarray],axis=-1)
                self.observation.append(array)
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                self.crtarray.fill(0)
                X,Y,Xl,Xu,Yl,Yu,Tt = self.square_dist(npc.rect.centerx,npc.rect.centery,self.map_grid)
                self.crtarray[X][Y]     += (Xu*Yu/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X+1][Y]   += (Xl*Yu/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X][Y+1]   += (Xu*Yl/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[0]/max_units).astype(int)
                self.crtarray[X+1][Y+1] += (Xl*Yl/Tt*max(0,(npc.hp_crt/npc.hp_max))*onehots[0]/max_units).astype(int)
                array = np.concatenate([allarray,self.crtarray],axis=-1)
                self.observation.append(array)
        self.observation = np.array(self.observation)
    def phase_action(self, action):
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j#k formula
                if action[k]==0: agt.vel_set(-self.mov_agt[i], 0)
                if action[k]==1: agt.vel_set( self.mov_agt[i], 0)
                if action[k]==2: agt.vel_set( 0,-self.mov_agt[i])
                if action[k]==3: agt.vel_set( 0, self.mov_agt[i])
                if action[k]==4: agt.vel_set( 0, 0)

        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                k = sum(self.num_npc[:i])+j#k formula
                k = k + sum(self.num_agt)
                if action[k]==0: npc.vel_set(-self.mov_npc[i], 0)
                if action[k]==1: npc.vel_set( self.mov_npc[i], 0)
                if action[k]==2: npc.vel_set( 0,-self.mov_npc[i])
                if action[k]==3: npc.vel_set( 0, self.mov_npc[i])
                if action[k]==4: #npc.vel_set( 0, 0)
                #continue
                    if npc.hp_crt<=0: continue
                    min_i,min_j = 0,0
                    min_hp = max(self.fhp_agt)+1
                    for index in self.atkorder:
                        i_index,j_index=0,0
                        inum=index
                        for ispecies,num_agt in enumerate(self.num_agt):
                            if inum>=num_agt:
                                inum=inum-num_agt
                            else:
                                i_index=ispecies
                                j_index=inum
                                break
                        if self.agts_list[i_index].sprites()[j_index].hp_crt<=0: continue
                        if self.agts_list[i_index].sprites()[j_index].hp_crt < min_hp:
                            min_i,min_j = i_index,j_index
                            min_hp= self.agts_list[i_index].sprites()[j_index].hp_crt
                    tx, ty = self.agts_list[min_i].sprites()[min_j].rect.centerx, self.agts_list[min_i].sprites()[min_j].rect.centery
                    sx, sy = npc.rect.centerx, npc.rect.centery
                    disst  = np.sqrt((tx-sx)*(tx-sx)+(ty-sy)*(ty-sy)) # np.sign
                    mx, my = (tx-sx)/disst/self.mov_npc[i], (ty-sy)/disst/self.mov_npc[i]
                    npc.vel_set(mx,my)
    def phase_passive(self):
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                agt.update()
                agt.pos_clip(0,self.map_width-1,0,self.map_height-1)
                npcs_overlap_list = []
                for i2,num_npc2 in enumerate(self.num_npc):
                    npcs_overlap_listi2 = pygame.sprite.spritecollide(agt, self.npcs_active_list[i2], False, pygame.sprite.collide_rect)
                    npcs_overlap_list.extend(npcs_overlap_listi2)
                for npc_overlap in npcs_overlap_list:
                    agt.stepback()
                    break
                agts_overlap_list = []
                for i2,num_agt2 in enumerate(self.num_agt):
                    agts_overlap_listi2 = pygame.sprite.spritecollide(agt, self.agts_active_list[i2], False, pygame.sprite.collide_rect)
                    agts_overlap_list.extend(agts_overlap_listi2)
                for agt_overlap in agts_overlap_list:
                    if agt_overlap == agt: continue
                    agt.stepback()
                    break
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                npc.update()
                npc.pos_clip(0,self.map_width-1,0,self.map_height-1)
                agts_overlap_list = []
                for i2,num_agt2 in enumerate(self.num_agt):
                    agts_overlap_listi2 = pygame.sprite.spritecollide(npc, self.agts_active_list[i2], False, pygame.sprite.collide_rect)
                    agts_overlap_list.extend(agts_overlap_listi2)
                for agt_overlap in agts_overlap_list:
                    npc.stepback()
                    break
                npcs_overlap_list = []
                for i2,num_npc2 in enumerate(self.num_npc):
                    npcs_overlap_listi2 = pygame.sprite.spritecollide(npc, self.npcs_active_list[i2], False, pygame.sprite.collide_rect)
                    npcs_overlap_list.extend(npcs_overlap_listi2)
                for npc_overlap in npcs_overlap_list:
                    if npc_overlap == npc: continue
                    npc.stepback()
                    break
    def phase_reward(self):
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                agt.hitten_list.clear()
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                npc.hitten_list.clear()
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                npcs_inrng = []
                for i2,num_npc2 in enumerate(self.num_npc):
                    npcs_inrngi2 = pygame.sprite.spritecollide(agt, self.npcs_active_list[i2], False, hit_collision)
                    npcs_inrng.extend(npcs_inrngi2)
                if len(npcs_inrng)==0: continue
                min_i  = 0#random.randrange(len(npcs_inrng))######################## if no random training failed ... why?
                min_hp = max(self.fhp_npc)+1
                for i,npc in enumerate(npcs_inrng):
                    if npc.hp_crt<=0: continue
                    if npc.hp_crt < min_hp:
                        min_i  = i
                        min_hp = npc.hp_crt
                if npcs_inrng[min_i].hp_crt<=0: continue
                npcs_inrng[min_i].hp_crt-=agt.atk
                npcs_inrng[min_i].hitten_list.append(agt.inum)
                if self.turn%2 == 0:
                    self.atklines.append([agt.color,agt.rect.centerx,agt.rect.centery,npcs_inrng[min_i].rect.centerx,npcs_inrng[min_i].rect.centery])
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                agts_inrng = []
                for i2,num_agt2 in enumerate(self.num_agt):
                    agts_inrngi2 = pygame.sprite.spritecollide(npc, self.agts_active_list[i2], False, hit_collision)#True to remove
                    agts_inrng.extend(agts_inrngi2)
                if len(agts_inrng)==0: continue
                min_i  = 0
                min_hp = max(self.fhp_agt)+1
                for i,agt in enumerate(agts_inrng):
                    if agt.hp_crt<=0: continue
                    if agt.hp_crt < min_hp:
                        min_i  = i
                        min_hp = agt.hp_crt
                if agts_inrng[min_i].hp_crt<=0: continue
                agts_inrng[min_i].hp_crt-=npc.atk
                agts_inrng[min_i].hitten_list.append(npc.inum)
                if self.turn%2 == 1:
                    self.atklines.append([npc.color,npc.rect.centerx,npc.rect.centery,agts_inrng[min_i].rect.centerx,agts_inrng[min_i].rect.centery])
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                color=np.array(agt.color)*max(0,(agt.hp_crt/agt.hp_max))
                color=color.astype(int)
                agt.image.fill(color)
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                color=np.array(npc.color)*max(0,(npc.hp_crt/npc.hp_max))
                color=color.astype(int)
                npc.image.fill(color)
    def phase_generate(self):
        self.epsd += 1
        agtloss = np.zeros(sum(self.num_agt))
        npcloss = np.zeros(sum(self.num_npc))
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j
                agtloss[k]=agt.loss
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                k = sum(self.num_npc[:i])+j
                npcloss[k]=npc.loss
        self.map_generate()
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j
                agt.loss=agtloss[k]
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                k = sum(self.num_npc[:i])+j
                npc.loss=npcloss[k]
    def phase_end(self):
        self.reward += self.reward_range[0]/self.end_turn
        if self.turn >= self.end_turn:
            self.done = True
            if self.epsd == 0:
                self.info['exinfos'] = -0.051
            else:
                if sum(self.learnflag[:len(self.num_agt)])<=sum(self.learnflag[len(self.num_agt):]):
                    self.info['exinfos'] = round(self.gain/self.epsd,3)
                else:
                    self.info['exinfos'] = round(self.loss/self.epsd,3)
            self.info['epsd'] = self.epsd
            self.info['gain'] = self.gain
            self.info['miss'] = self.miss
            self.info['loss'] = self.loss
            return
        if self.repeat:
            self.reward += self.reward_range[-3]
            self.miss+=1
            self.phase_generate()
            return
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j
                agt.gain = agt.hp_crt
                if not self.agts_active_list[i].has(agt): continue
                if agt.hp_crt<=0:
                    self.reward[k] += self.reward_range[2]
                    agt.loss+=1
                    self.agts_active_list[i].remove(agt)
                    for ihitter in agt.hitten_list:
                        self.reward[ihitter+sum(self.num_agt)] += self.reward_range[1]
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                k = sum(self.num_npc[:i])+j
                npc.gain = npc.hp_crt
                if not self.npcs_active_list[i].has(npc): continue
                if npc.hp_crt<=0:
                    self.reward[k+sum(self.num_agt)] += self.reward_range[2]
                    npc.loss+=1
                    self.npcs_active_list[i].remove(npc)
                    for ihitter in npc.hitten_list:
                        self.reward[ihitter] += self.reward_range[1]

        unitlosses = []
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j
                if self.agts_active_list[i].has(agt): unitlosses.append(False)
                else:                                 unitlosses.append(True)
        for i,num_agt in enumerate(self.num_agt):
            speciesloss = all(unitlosses[sum(self.num_agt[:i]):sum(self.num_agt[:i+1])])
            if speciesloss==True and self.speciesloss[i]==False:
                self.reward[sum(self.num_agt[:i]):sum(self.num_agt[:i+1])]+=self.reward_range[-2]
                self.speciesloss[i]=True

        if sum([len(self.agts_active_list[i]) for i,num_agt in enumerate(self.num_agt)])==0:
            self.reward[sum(self.num_agt):]+=self.reward_range[-1] ###
            self.loss+=1
            self.phase_generate()
            return
        if sum([len(self.npcs_active_list[i]) for i,num_npc in enumerate(self.num_npc)])==0:
            self.reward[sum(self.num_agt):]+=self.reward_range[-2] ###
            for i,num_agt in enumerate(self.num_agt):
                if self.speciesloss[i]==False: self.reward[sum(self.num_agt[:i]):sum(self.num_agt[:i+1])]+=self.reward_range[-1]
            self.gain+=1
            self.phase_generate()
            return
    def step(self, action):
        self.model_nums = action[-(len(self.num_agt)+len(self.num_npc)):]
        action = action[:-(len(self.num_agt)+len(self.num_npc))]
        self.atklines.clear()
        self.reward = np.zeros(sum(self.num_agt)+sum(self.num_npc))
        self.phase_action(action)
        self.phase_passive()
        self.phase_reward()
        self.phase_end()
        self.turn+=1
        self.get_observation()
        return self.observation, self.reward, self.done, self.info

    def render(self, mode='rgb_array', close=False):
        screen_small_obs = []
        for i,oobs in enumerate(self.observation):
            obs=[]
            for ioobs in oobs:
                iobs=[]
                for ijoobs in ioobs:
                    v = np.sum((ijoobs+0)*np.array([pow(255,i) for i in range(self.layernum)]))
                    cv = v/pow(255,self.layernum)
                    cv = abs(cv)
                    cvv = pow(cv,self.scale)
                    c=self.mapper.to_rgba(cvv)
                    cc=np.array(c)*255
                    ccc=cc.astype(int)[:-1]
                    iobs.append(ccc)
                obs.append(iobs)
            obs = np.array(obs)
            screen_small_obs.append(obs)
        #print(self.observation.shape)
        #for obs in screen_small_obs:
        #    print(obs.shape)
        #exit()
        self.screen_small = [pygame.surfarray.make_surface(obs) for obs in screen_small_obs]
        #for iscreen,screen_small in enumerate(self.screen_small):
        #    self.real_screen.blit(pygame.transform.scale(screen_small, [self.real_screen_width, self.real_screen_height]), (self.real_screen_width*(iscreen+1), 0))
        for i in range(sum(self.num_agt)):
            screen_small = pygame.transform.scale(self.screen_small[i], [self.real_screen_width, self.real_screen_height])
            self.real_screen.blit(screen_small, (self.real_screen_width*i, self.real_screen_height*1))
        for i in range(sum(self.num_npc)):
            screen_small = pygame.transform.scale(self.screen_small[i+sum(self.num_agt)], [self.real_screen_width, self.real_screen_height])
            self.real_screen.blit(screen_small, (self.real_screen_width*i, self.real_screen_height*2))
        for i in range(1,self.num_screen[1]):
            pygame.draw.line(self.real_screen, DGRAY, (0, self.real_screen_height*i), (self.real_screen_width*self.num_screen[0], self.real_screen_height*i))
        for i in range(1,self.num_screen[0]):
            pygame.draw.line(self.real_screen, DGRAY, (self.real_screen_width*i, 0), (self.real_screen_width*i, self.real_screen_height*self.num_screen[1]))
        self.screen.fill(BLACK)
        for listi in self.lists_visible:
            listi.draw(self.screen)
        self.real_screen.blit(pygame.transform.scale(self.screen, [self.real_screen_width, self.real_screen_height]), (0, 0))
        for i in range(0,self.real_screen_width, int(self.times)):
            pygame.draw.line(self.real_screen, DGRAY, (i, 0), (i, self.real_screen_height))
        for j in range(0,self.real_screen_height,int(self.times)):
            pygame.draw.line(self.real_screen, DGRAY, (0, j), (self.real_screen_width, j))
        for atkline in self.atklines:
            pygame.draw.line(self.real_screen, atkline[0], (atkline[1]*self.times+int(self.times/2),atkline[2]*self.times+int(self.times/2))
                                                         , (atkline[3]*self.times+int(self.times/2),atkline[4]*self.times+int(self.times/2)))
        fontsize, linespace, startline = self.times_small*self.map_height_small//30, self.times_small*self.map_height_small//30, 0
        fontposx = 0#self.real_screen_width
        myfont = pygame.font.SysFont("arial", fontsize)#arial # font8=6px
        for i,num_agt in enumerate(self.num_agt):
            model_num = myfont.render('M:%s' %str(self.model_nums[i]),True, BGRAY)
            self.real_screen.blit(model_num,(fontposx,linespace*0+startline))
            startline += linespace*1
        for i,num_npc in enumerate(self.num_npc):
            model_num = myfont.render('M:%s' %str(self.model_nums[i+len(self.num_agt)]),True, BGRAY)
            self.real_screen.blit(model_num,(fontposx,linespace*0+startline))
            startline += linespace*1
        gain = myfont.render('G:%s' %str(self.gain),True, BGRAY)
        loss = myfont.render('L:%s' %str(self.loss),True, BGRAY)
        miss = myfont.render('M:%s' %str(self.miss),True, BGRAY)
        epsd = myfont.render('E:%s' %str(self.epsd),True, BGRAY)
        turn = myfont.render('T:%s' %str(self.turn),True, BGRAY)
        self.real_screen.blit(gain,(fontposx,linespace*0+startline))
        self.real_screen.blit(loss,(fontposx,linespace*1+startline))
        self.real_screen.blit(miss,(fontposx,linespace*2+startline))
        self.real_screen.blit(epsd,(fontposx,linespace*3+startline))
        self.real_screen.blit(turn,(fontposx,linespace*4+startline))
        startline += linespace*6
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                gain = myfont.render('G:%s' %str(npc.gain),True, npc.color)
                loss = myfont.render('L:%s' %str(npc.loss),True, npc.color)
                self.real_screen.blit(gain,(fontposx,linespace*0+startline))
                self.real_screen.blit(loss,(fontposx,linespace*1+startline))
                startline += linespace*2
        startline += linespace*1
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                gain = myfont.render('G:%s' %str(agt.gain),True, agt.color)
                loss = myfont.render('L:%s' %str(agt.loss),True, agt.color)
                self.real_screen.blit(gain,(fontposx,linespace*0+startline))
                self.real_screen.blit(loss,(fontposx,linespace*1+startline))
                startline += linespace*2
        pygame.display.update() #pygame.display.flip()
        frame = pygame.surfarray.array3d(self.real_screen).swapaxes(0,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if mode=='human' and close==False: self.clock.tick(self.timelag)
        return frame
    def close(self):
        pass
    def seed(self, seed=None):
        random.seed(seed)
