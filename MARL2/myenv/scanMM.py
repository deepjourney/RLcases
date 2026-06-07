import numpy as np
import random, gym, gym.spaces, json, easydict, time, cv2, scipy
from itertools import cycle
import matplotlib
import matplotlib.cm as cm
from myenv import *
class ScanMM(gym.Env):
    def __init__(self):
        self.walls=[]
        lines = open('./myenv/setting.txt', 'r').read().splitlines()
        for line in lines:
            self.walls.append([int(data) for data in line.split(',')])
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
        self.map_width, self.map_height = self.map_width_small, self.map_height_small
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
        self.num_npc[0]+=int(self.switches[0])######
        self.num_npc.append(len(self.walls))
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
        self.npcarray = np.zeros((self.map_width_small,self.map_height_small,3),dtype=np.int8) #len(self.num_npc)),dtype=np.int8) # np.sum default is int64 not int8
        self.agtarray = np.zeros((self.map_width_small,self.map_height_small,3),dtype=np.int8) #len(self.num_agt)),dtype=np.int8)
        self.crtarray = np.zeros((self.map_width_small,self.map_height_small,1),dtype=np.int8) # int8 is from -128 to 127, good input data +/- balance for 255 colors
        self.layernum = self.npcarray.shape[-1]+self.agtarray.shape[-1]+self.crtarray.shape[-1]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        self.mapper, self.scale = cm.ScalarMappable(norm=norm, cmap=cm.nipy_spectral), 0.1#gist_ncar,bwr,nipy_spectral,gnuplot,gnuplot2,gist_earth,gist_stern,inferno,jet,hsv,cm.Greys_r)
        self.reset()
        if args.render: self.setsize(args.times,0)
        #print(':',time.time()-self.starttime)
        #self.starttime = time.time()
        self.observation_space = None#gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,self.map_channel],dtype=np.uint8)
        self.action_space      = None#gym.spaces.Discrete(self.act_agt)
        self.reward_range      = args.rew_rng ### should be no repeat element list?
        self.attr = {}
        self.attr['unit_nums']  = self.num_agt#[sum(self.num_agt)]
        self.attr['obs_spaces'] = [gym.spaces.Box(low=0,high=255,shape=[self.map_width_small,self.map_height_small,3],dtype=np.uint8)#self.layernum],dtype=np.uint8)
                                   for i,num_agt in enumerate(self.attr['unit_nums'])] #dtype=np.uint8
        self.attr['act_spaces'] = [gym.spaces.Discrete(self.act_agt[i]) \
                                   for i,num_agt in enumerate(self.attr['unit_nums'])]
    def __del__(self):
        pygame.quit()
    def setsize(self,times,timelag):
        self.real_screen_width, self.real_screen_height, self.times_small, self.timelag = self.map_width_small*times, self.map_height_small*times, times, timelag
        self.times = self.map_width_small*times//self.map_width
        self.video_size, self.num_screen = [self.map_width_small,self.map_height_small,3], (sum(self.num_agt)+1+1)
        self.real_screen = pygame.display.set_mode([self.real_screen_width*self.num_screen, self.real_screen_height]) ##,HWSURFACE|DOUBLEBUF|RESIZABLE
    def reset(self):
        self.atklines = []
        self.info, self.done, self.turn, self.epsd = {}, False, 0, 0
        self.gain, self.loss, self.miss = 0, 0, 0
        self.map_generate()
        self.get_observation()
        return self.observation
    def map_generate(self):
        self.atklines.clear()
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
        num_npc = sum(self.num_npc)-len(self.walls)#random.randrange(self.min_num_npc,self.max_num_npc+1)
        semiperimeter = 4#6#4#6#(self.map_width+self.map_height)//3
        edges = [2]#,3]#[1,2,3,6]#4]#,3,6]
        npclenx = [random.choice(edges) for i in range(num_npc)]
        npcleny = [semiperimeter//lenx for lenx in npclenx]
        npcplace = random.sample(range(1,self.map_width-1), num_npc)
        npcplacy = random.sample(range(1,self.map_height-1),num_npc)
        for i,num_npc in enumerate(self.num_npc):
            if i==len(self.num_npc)-1:
                for j,wall in enumerate(self.walls):
                    k = sum(self.num_npc[:i])+j#k formula
                    npc = LPVSprite(ispecies=i,inum=k,color=WHITE,lenx=wall[2]-wall[0]+1,leny=wall[3]-wall[1]+1,posx=wall[0]+(wall[2]-wall[0]+1)//2,posy=wall[1]+(wall[3]-wall[1]+1)//2)
                    self.npcs_list[i].add(npc)
                    if npc.hp_crt>0: self.npcs_active_list[i].add(npc)
                continue
            for j in range(num_npc):
                k = sum(self.num_npc[:i])+j#k formula
                color = pygame.color.THECOLORS[next(colorbar4)]
                npc = LPVSprite(ispecies=i,inum=k,color=WHITE,lenx=npclenx[k],leny=npcleny[k],posx=npcplace[k],posy=npcplacy[k],hp_max=self.fhp_npc[i],hp_crt=self.fhp_npc[i],atk=self.atk_npc[i],rng=self.rng_npc[i])
                npc.pos_clip(0,self.map_width-1,1,self.map_height-1) ### skip top line for agent initial place
                self.npcs_list[i].add(npc)
                if npc.hp_crt>0: self.npcs_active_list[i].add(npc)
        num_agt = sum(self.num_agt)
        agtplacy = np.zeros(num_agt)
        agtplace = random.sample(range(self.map_width//3), num_agt)
        agtplace = np.array(agtplace)*3
        for i,num_agt in enumerate(self.num_agt):
            agthpcrt = []
            for j in range(num_agt):
                k = sum(self.num_agt[:i])+j#k formula
                color = pygame.color.THECOLORS[next(colorbar1)]######switches
                agt = LPVSprite(ispecies=i,inum=k,color=color,lenx=1+int(self.switches[i+1]),leny=1+int(self.switches[i+1]),posx=agtplace[k],posy=agtplacy[k],hp_max=self.fhp_agt[i],hp_crt=self.fhp_agt[i],atk=self.atk_agt[i],rng=self.rng_agt[i])
                self.agts_list[i].add(agt)
                if agt.hp_crt>0: self.agts_active_list[i].add(agt)

        self.bcgscreen = pygame.Surface([self.map_width, self.map_height])
        self.bcgscreen.fill(DGRAY)
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                npc.image.fill(BLACK)
            self.npcs_active_list[i].draw(self.bcgscreen)
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                npc.image.fill(npc.color)

        self.bcgarray = np.ones([self.map_width, self.map_height])
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_active_list[i]):
                x,y,lx,ly = npc.rect.x, npc.rect.y, npc.lenx, npc.leny
                for j in range(x,x+lx):
                    for k in range(y,y+ly):
                        self.bcgarray[j][k]=0
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                x,y,lx,ly = agt.rect.x, agt.rect.y, agt.lenx, agt.leny
                for j in range(x,x+lx):
                    for k in range(y,y+ly):
                        if j<0 or k<0: continue
                        self.bcgarray[j][k]=0
        self.fulbcg = np.sum(self.bcgarray)
        self.crtbcg = self.fulbcg
        self.prebcg = self.crtbcg

        self.trigger = 10
        for i,num_npc in enumerate(self.num_npc):
            if i==len(self.num_npc)-1: continue
            for j,npc in enumerate(self.npcs_list[i]):
                npc_action = random.randrange(4)
                if npc_action==0: npc.vel_set(0, 1.0/self.mov_npc[i])
                if npc_action==1: npc.vel_set(0,-1.0/self.mov_npc[i])
                if npc_action==2: npc.vel_set( 1.0/self.mov_npc[i],0)
                if npc_action==3: npc.vel_set(-1.0/self.mov_npc[i],0)
    def get_observation(self):
        self.screen_small.clear()
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                agt.image.fill(BLACK)
            self.agts_active_list[i].draw(self.bcgscreen)
        agtcolor,maincolor = [GREEN,BLUE,YELLOW],RED
        observation = []
        common_screen = self.bcgscreen.copy()
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                agt.image.fill(agtcolor[i])
            self.agts_active_list[i].draw(common_screen)
        for i,num_npc in enumerate(self.num_npc):
            self.npcs_active_list[i].draw(common_screen)
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                screen = common_screen.copy()
                agt.image.fill(maincolor)
                self.agts_active_list[i].draw(screen)#pygame.sprite.Group(agt).draw(screen)
                observation.append(pygame.surfarray.array3d(screen))
                self.screen_small.append(screen)
                agt.image.fill(agtcolor[i])
        self.observation = np.array(observation)
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                agt.image.fill(agt.color)
    def phase_action(self, action):
        #action = [random.randrange(4) for i in range(sum(self.num_agt))]
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                k = sum(self.num_agt[:i])+j#k formula
                if action[k]==0: agt.vel_set(-1.0/self.mov_agt[i], 0)
                if action[k]==1: agt.vel_set( 1.0/self.mov_agt[i], 0)
                if action[k]==2: agt.vel_set( 0,-1.0/self.mov_agt[i])
                if action[k]==3: agt.vel_set( 0, 1.0/self.mov_agt[i])
                if action[k]==4: agt.vel_set( 0, 0)
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
                self.atklines.append([agt.color,agt.rect.centerx,agt.rect.centery,int(agt.pposx),int(agt.pposy),agt.inum*2])

        for i,num_npc in enumerate(self.num_npc):
            if i==len(self.num_npc)-1: continue
            for j,npc in enumerate(self.npcs_active_list[i]):
                npc.update()
                agts_overlap_list = []
                for i2,num_agt2 in enumerate(self.num_agt):
                    agts_overlap_listi2 = pygame.sprite.spritecollide(npc, self.agts_active_list[i2], False, pygame.sprite.collide_rect)
                    agts_overlap_list.extend(agts_overlap_listi2)
                #agts_overlap_list = pygame.sprite.spritecollide(npc, self.agts_active_list, False, pygame.sprite.collide_rect)#True to remove
                for agt_overlap in agts_overlap_list:
                    npc.stepback()
                    break
                npcs_overlap_list = []
                for i2,num_npc2 in enumerate(self.num_npc):
                    npcs_overlap_listi2 = pygame.sprite.spritecollide(npc, self.npcs_active_list[i2], False, pygame.sprite.collide_rect)
                    npcs_overlap_list.extend(npcs_overlap_listi2)
                for npc_overlap in npcs_overlap_list:
                    if npc_overlap==npc: continue
                    npc.stepback()
                    break
                clipflag = npc.pos_clip(0,self.map_width-1,0,self.map_height-1)
                if clipflag or self.turn%self.trigger==0:
                    npc_action = random.randrange(4)
                    if npc_action==0: npc.vel_set(0, 1.0/self.mov_npc[i])
                    if npc_action==1: npc.vel_set(0,-1.0/self.mov_npc[i])
                    if npc_action==2: npc.vel_set( 1.0/self.mov_npc[i],0)
                    if npc_action==3: npc.vel_set(-1.0/self.mov_npc[i],0)
    def phase_end(self):
        self.reward += self.reward_range[0]/self.end_turn

        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_active_list[i]):
                k = sum(self.num_agt[:i])+j#k formula
                x,y,lx,ly,nfill = agt.rect.x, agt.rect.y, agt.lenx, agt.leny, 0
                for m in range(x,x+lx):
                    for n in range(y,y+ly):
                        if self.bcgarray[m][n]!=0:
                            self.bcgarray[m][n]=0
                            nfill+=1
                self.reward[k]+=self.reward_range[1]*nfill
                agt.gain+=1*nfill
                agt.loss=self.reward[k]
        self.crtbcg = np.sum(self.bcgarray)

        self.gain = self.fulbcg-self.crtbcg
        self.loss += np.sum(self.reward)
        self.miss = self.prebcg-self.crtbcg
        self.prebcg = self.crtbcg
        if self.crtbcg == 0:
            self.reward += self.reward_range[-1]
            self.epsd+=1
            #maxgain = -1
            #maxid   = -1
            #for i,num_agt in enumerate(self.num_agt):
            #    for j,agt in enumerate(self.agts_active_list[i]):
            #        if agt.gain>maxgain:
            #            maxid = i
            #            maxgain = agt.gain
            #self.reward[maxid]+=self.reward_range[-2]
            self.done=True
            self.info['exinfos'] = round((self.fulbcg)/(self.turn),3)
        if self.turn >= self.end_turn:
            if self.epsd==0:
                self.reward += self.reward_range[-3]
            self.done = True
            self.info['exinfos'] = round((self.fulbcg*(self.epsd+1)-self.crtbcg)/(self.turn),3)
    def step(self, action):
        self.reward = np.zeros(sum(self.num_agt))
        self.phase_action(action)
        self.phase_passive()
        self.phase_end()
        self.turn+=1
        self.get_observation()
        return self.observation, self.reward, self.done, self.info

    def render(self, mode='rgb_array', close=False):
        framearrays = np.expand_dims(self.bcgarray,-1)*DGRAY[:3]
        framescreen = pygame.surfarray.make_surface(framearrays)
        self.real_screen.blit(pygame.transform.scale(framescreen, [self.real_screen_width, self.real_screen_height]), (self.real_screen_width*(sum(self.num_agt)+1), 0))
        for iscreen,screen_small in enumerate(self.screen_small):
            self.real_screen.blit(pygame.transform.scale(screen_small, [self.real_screen_width, self.real_screen_height]), (self.real_screen_width*(iscreen+1), 0))
        for i in range(0,self.real_screen_width*(sum(self.num_agt)+1), int(self.real_screen_width)):
            pygame.draw.line(self.real_screen, GRAY, (i, 0), (i, self.real_screen_height))
        self.screen = self.bcgscreen.copy()#.fill(BLACK)
        for listi in self.lists_visible:
            listi.draw(self.screen)
        self.real_screen.blit(pygame.transform.scale(self.screen, [self.real_screen_width, self.real_screen_height]), (0, 0))
        for i in range(0,self.real_screen_width, int(self.times)):
            pygame.draw.line(self.real_screen, GRAY, (i, 0), (i, self.real_screen_height))
        for j in range(0,self.real_screen_height,int(self.times)):
            pygame.draw.line(self.real_screen, GRAY, (0, j), (self.real_screen_width, j))
        for atkline in self.atklines:
            pygame.draw.line(self.real_screen, atkline[0], (atkline[1]*self.times+int(self.times/2)+atkline[-1],atkline[2]*self.times+int(self.times/2)+atkline[-1])
                                                         , (atkline[3]*self.times+int(self.times/2)+atkline[-1],atkline[4]*self.times+int(self.times/2)+atkline[-1]))
        fontsize, linespace, startline = self.times_small*self.map_height_small//30, self.times_small*self.map_height_small//30, 0
        myfont = pygame.font.SysFont("arial", fontsize)#arial # font8=6px
        gain = myfont.render('G:%s' %str(self.gain),True, BGRAY)
        loss = myfont.render('L:%s' %str(self.loss),True, BGRAY)
        miss = myfont.render('M:%s' %str(self.miss),True, BGRAY)
        epsd = myfont.render('E:%s' %str(self.epsd),True, BGRAY)
        turn = myfont.render('T:%s' %str(self.turn),True, BGRAY)
        self.real_screen.blit(gain,(0,linespace*0+startline))
        self.real_screen.blit(loss,(0,linespace*1+startline))
        self.real_screen.blit(miss,(0,linespace*2+startline))
        self.real_screen.blit(epsd,(0,linespace*3+startline))
        self.real_screen.blit(turn,(0,linespace*4+startline))
        startline += linespace*6
        for i,num_npc in enumerate(self.num_npc):
            for j,npc in enumerate(self.npcs_list[i]):
                gain = myfont.render('G:%s' %str(npc.gain),True, npc.color)
                loss = myfont.render('L:%s' %str(npc.loss),True, npc.color)
                self.real_screen.blit(gain,(0,linespace*0+startline))
                self.real_screen.blit(loss,(0,linespace*1+startline))
                startline += linespace*2
        startline += linespace*1
        for i,num_agt in enumerate(self.num_agt):
            for j,agt in enumerate(self.agts_list[i]):
                gain = myfont.render('G:%s' %str(agt.gain),True, agt.color)
                loss = myfont.render('L:%s' %str(agt.loss),True, agt.color)
                self.real_screen.blit(gain,(0,linespace*0+startline))
                self.real_screen.blit(loss,(0,linespace*1+startline))
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
