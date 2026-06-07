import numpy as np
import random, gym, gym.spaces, json, easydict, time, sys, cv2, copy
from pprint import pprint
from myenv import *
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
HALITE,YARDS,SHIPS,POS,CARGO = 0,1,2,0,1
class HaliteM(gym.Env):
    def __init__(self):
        super().__init__()
        with open('./myenv/envinfo.json', 'r') as envinfo_file:
            envinfo_args_dict = easydict.EasyDict(json.load(envinfo_file))
        args = envinfo_args_dict
        self.play_num = args.play_num
        self.type_num = args.type_num
        self.unit_num = args.unit_num
        self.learnflag= []
        for learnflags in args.learnflag.split('_'):
            self.learnflag.append([int(learnflag) for learnflag in learnflags.split(',')])
        envparas       = args.envparas.split('_')
        game_setting   = envparas[0].split(',')
        self.boardsize    = int(game_setting[0])
        self.episodeSteps = int(game_setting[1])
        self.initial_cash = [int(game_setting[2])]*self.play_num
        self.maxHold, self.maxCargo = 10000, 10000
        self.envH = make("halite", configuration={"size":self.boardsize, "episodeSteps":self.episodeSteps}, debug=True)
        #"startingHalite":3000 #,'spawnCost':50, 'convertCost':50
        if args.render:
            self.envH.renderer = myrenderer
            pygame.init()
            pygame.mixer.quit()
            self.video_size = [400+80*self.boardsize, 400+80*self.boardsize, 3]
            self.real_screen = pygame.display.set_mode(self.video_size[:2])
        reward_setting = envparas[1].split(',')
        self.rewardratio  = float(reward_setting[0])
        self.timecost     = float(reward_setting[1])
        self.yardshare    = float(reward_setting[2])
        envonoff       = args.envonoff.split(',')
        self.shuffleon    = bool(int(envonoff[0]))
        self.rulemoveon   = bool(int(envonoff[1]))
        #pprint(self.envH.specification)
        #pprint(self.envH.configuration)#exit()#self.ship_layer, self.yard_layer, 
        self.cargo_layer, self.hold_layer, self.layers_each = 0,1,2#,3,4#self.marker_layer, self.step_layer, 
        self.halite_layer, self.layers_whole = 0,1#,2#,3
        self.channel_num = self.play_num*self.layers_each+self.layers_whole +1#self.unit_num
        observation_shape = [self.play_num,self.type_num,self.unit_num,self.boardsize,self.boardsize,self.channel_num]
        self.observation_space = gym.spaces.Box(low=0,high=255,shape=observation_shape,dtype=np.uint8)
        self.action_space      = None#gym.spaces.Discrete(self.act_agt)
        self.reward_range      = [0,1]
        self.reset()
        self.attr = {}
    def reset(self):
        self.envH.reset(self.play_num)
        self.obs = self.envH.state[0].observation
        for i in range(self.play_num):
            self.obs['players'][i][HALITE]=self.initial_cash[i]
        self.obs['config']=self.envH.configuration
        self.players_crt = self.envH.state[0].observation['players']
        self.players_pre = self.envH.state[0].observation['players']
        self.player_order = [i for i in range(self.play_num)]
        if self.shuffleon: random.shuffle(self.player_order)
        #pprint(self.states)#print(Board(self.obs,self.obs['config']))
        #pprint(self.obs)#exit()
        self.reward = np.zeros([self.play_num,self.type_num,self.unit_num])
        self.rewall = np.zeros([self.play_num,self.type_num,self.unit_num])
        self.initall= sum(self.obs['halite'])
        self.actions, self.actions_string = np.zeros([self.play_num,self.type_num,self.unit_num]), [{}]*self.play_num
        self.shipbuild, self.yardbuild = [0]*self.play_num, [0]*self.play_num
        self.ship_states, self.info = [{}]*self.play_num, {}
        arrayobs = self.get_arrayobs(self.obs)
        return arrayobs
    def get_arrayobs_i(self,obs,conf,crt_playernum):
        if crt_playernum==0: player_order,k = [0,1,2,3],0
        if crt_playernum==1: player_order,k = [1,0,3,2],1
        if crt_playernum==2: player_order,k = [2,3,0,1],3
        if crt_playernum==3: player_order,k = [3,2,1,0],2
        playermap = np.zeros([self.channel_num,len(obs['halite'])],dtype=np.uint8)
        for iplayer,playernum in enumerate(player_order):
            for inum,yardid_yardvalue in enumerate(list(obs['players'][playernum][YARDS].items())):
                yardid, yardvalue = yardid_yardvalue
                playermap[iplayer*self.layers_each+self.hold_layer][yardvalue] = 105+min(150,obs['players'][playernum][HALITE]*150/self.maxHold)
            for inum,shipid_shipvalue in enumerate(list(obs['players'][playernum][SHIPS].items())):
                shipid, shipvalue = shipid_shipvalue
                playermap[iplayer*self.layers_each+self.cargo_layer][shipvalue[POS]]= 105+min(150,shipvalue[CARGO]*150/self.maxCargo)
        playermap[self.play_num*self.layers_each+self.halite_layer] = np.array(obs['halite'])*255/conf.maxCellHalite
        #playermap[self.play_num*self.layers_each+self.step_layer]   = np.ones(len(obs['halite']))*obs['step']*255/conf.episodeSteps
        # self.channel_num, boardsize*boardsize
        playermap = [playermap for i in range(self.unit_num)]
        playermap = np.stack(playermap, axis=0)
        playermap = [playermap for i in range(self.type_num)]
        playermap = np.stack(playermap, axis=0)
        itype=0
        for inum,shipid_shipvalue in enumerate(list(obs['players'][crt_playernum][SHIPS].items())):
            shipid, shipvalue = shipid_shipvalue
            playermap[itype][inum][-1][shipvalue[POS]] = 255
        playermap = np.transpose(playermap,(0,1,3,2))
        playermap = playermap.reshape(self.type_num,self.unit_num,self.boardsize,self.boardsize,self.channel_num)
        playermap_r = np.zeros([self.type_num,self.unit_num,self.boardsize,self.boardsize,self.channel_num],dtype=np.uint8)
        for itype in range(self.type_num):
            for iunit in range(self.unit_num):
                if crt_playernum==0: playermap_r[itype][iunit] = playermap[itype][iunit]
                if crt_playernum==1: playermap_r[itype][iunit] = np.fliplr(playermap[itype][iunit])
                if crt_playernum==2: playermap_r[itype][iunit] = np.flipud(playermap[itype][iunit])
                if crt_playernum==3: playermap_r[itype][iunit] = np.fliplr(np.flipud(playermap[itype][iunit]))
        return playermap_r
    def get_arrayobs(self,obs):
        playermaps = np.zeros([self.play_num,self.type_num,self.unit_num,self.boardsize,self.boardsize,self.channel_num],dtype=np.uint8)
        for iplayer,playernum in enumerate(self.player_order):
            playermaps[iplayer] = self.get_arrayobs_i(obs,obs['config'],playernum)
        self.playermap = playermaps[0][0][0]
        return playermaps
    def get_stringact_i(self,actionsi,obs,crt_playernum):
        if crt_playernum==0: actionset = ['NORTH','EAST','SOUTH','WEST']
        if crt_playernum==1: actionset = ['NORTH','WEST','SOUTH','EAST']
        if crt_playernum==2: actionset = ['SOUTH','EAST','NORTH','WEST']
        if crt_playernum==3: actionset = ['SOUTH','WEST','NORTH','EAST']
        playernum = crt_playernum
        itype = 0
        stringact = {}
        for inum,shipid_shipvalue in enumerate(list(obs['players'][playernum][SHIPS].items())):
            shipid, shipvalue = shipid_shipvalue
            if actionsi[itype][inum]>0:
                stringact[shipid] = actionset[4-actionsi[itype][inum]]
            if shipvalue[CARGO] >= 5000: ######
                stringact[shipid] = 'CONVERT'
        if len(obs['players'][playernum][YARDS].items())<1:
            for inum,shipid_shipvalue in enumerate(list(obs['players'][playernum][SHIPS].items())):
                shipid, shipvalue = shipid_shipvalue
                if shipvalue[CARGO]>=500 or obs['players'][playernum][HALITE]>=500:
                    stringact[shipid] = 'CONVERT'
                    break # if all ship become yard together, there will be several yards and they will produce several ships at once next time and over the unit_num limit
        for inum,yardid_yardvalue in enumerate(list(obs['players'][playernum][YARDS].items())):
            if inum!=len(obs['players'][playernum][YARDS].items())-1: continue
            yardid, yardvalue = yardid_yardvalue ######
            if len(obs['players'][playernum][SHIPS].items())<self.unit_num:
                stringact[yardid] = 'SPAWN'
        return stringact
    def get_stringact(self,actions):
        self.actions = actions
        actions_string = [{}]*self.play_num
        for iplayer,playernum in enumerate(self.player_order):
            itype=0
            if self.learnflag[iplayer][itype]==-1:
                stringact = self.ruleagent(self.obs,self.obs['config'],playernum)
            else:
                stringact = self.get_stringact_i(actions[iplayer],self.obs,playernum)
            actions_string[playernum] = stringact
        self.actions_string = actions_string
        return actions_string



    def cal_asset(self, player):
        money,shipyard,ship = player[0],player[1],player[2]
        asset = money+self.envH.configuration.spawnCost*(len(shipyard.items())+len(ship.items()))+self.envH.configuration.convertCost*len(shipyard.items())
        for key,value in ship.items():
            asset+=value[1]
        return asset
    def get_reward(self, players_pre, players_crt):
        reward = np.zeros([self.play_num,self.type_num,self.unit_num])
        for iplayer,playernum in enumerate(self.player_order):
            for iyardnum,yardid_yardvalue in enumerate(list(players_crt[playernum][YARDS].items())):
                yardid, yardvalue = yardid_yardvalue
                if yardid not in players_pre[playernum][YARDS]:
                    self.yardbuild[playernum]+=1
            for ishipnum,shipid_shipvalue in enumerate(list(players_crt[playernum][SHIPS].items())):
                shipid, shipvalue = shipid_shipvalue
                if shipid not in players_pre[playernum][SHIPS]:
                    self.shipbuild[playernum]+=1
            itype=0
            if self.learnflag[iplayer][itype]!=0: continue ######
            reward[iplayer][itype] += self.timecost ######
            for iyardnum,yardid_yardvalue in enumerate(list(players_pre[playernum][YARDS].items())):
                yardid, yardvalue = yardid_yardvalue
            for ishipnum,shipid_shipvalue in enumerate(list(players_pre[playernum][SHIPS].items())):
                shipid, shipvalue = shipid_shipvalue
                if shipid in players_crt[playernum][SHIPS]:
                    backyardflag = False
                    for jyardnum,yardid_yardvalue in enumerate(list(players_crt[playernum][YARDS].items())):
                        yardid, yardvalue = yardid_yardvalue
                        if players_crt[playernum][SHIPS][shipid][POS]==yardvalue:
                            backyardflag = True
                            break
                    if backyardflag:
                        reward[iplayer][itype][ishipnum] += shipvalue[CARGO]*self.yardshare ######
                    else:
                        reward[iplayer][itype][ishipnum] += (players_crt[playernum][SHIPS][shipid][CARGO]-shipvalue[CARGO])*(1-self.yardshare) ######
                else:
                    reward[iplayer][itype][ishipnum] += -500-shipvalue[CARGO]
                    if shipid in self.actions_string[playernum]:
                            if self.actions_string[playernum][shipid]=='CONVERT':
                                reward[iplayer][itype][ishipnum] += 500+shipvalue[CARGO]

        mean = np.mean(reward,axis=-1)
        mean = [mean for i in range(self.unit_num)]
        mean = np.stack(mean, axis=-1)
        reward += mean
        reward = reward/2
        self.reward = reward
        self.rewall+= reward
        if 0:#self.envH.done:
            loses = []
            for iplayer,playernum in enumerate(self.player_order):
                loses.append(self.check_defeat(players[i]))
            richestplayer = np.argmax(np.array([players[i][HALITE] for i in range(self.play_num)]))
            self.reward = np.ones([self.play_num,self.type_num,self.unit_num])*(-1000)
            if not loses[richestplayer]: self.reward[richestplayer] = np.ones([self.type_num,self.unit_num])*(1000)
            if not loses[richestplayer] and self.learnflag[richestplayer][0]==0: self.info['exinfos']=1 ######
            else:                                                                self.info['exinfos']=0
        reward = reward*self.rewardratio#np.array([reward[playernum] for playernum in self.player_order])*self.rewardratio
        return reward
    def check_defeat(self,player):
        if len(player[2].items()) == 0 and (len(player[1].items()) == 0 or player[0] < self.envH.configuration.spawnCost): return True
        else: return False
    def step(self, actions):
        stringact = self.get_stringact(actions)
        states = self.envH.step(stringact)
        self.obs = states[0].observation
        self.obs['config']=self.envH.configuration
        self.players_crt = self.envH.state[0].observation['players']
        arrayrew = self.get_reward(self.players_pre,self.players_crt)
        self.players_pre = self.envH.state[0].observation['players']#copy.deepcopy(self.envH.state[0].observation['players'])
        arrayobs = self.get_arrayobs(self.obs)
        if 0:
            #board = Board(self.obs,self.obs['config'])
            #print(states)
            #print(actions)
            #print(self.actions_string)
            #print(self.reward)
            #print(players)
            """with np.printoptions(threshold=np.inf):
                for i0 in range(self.play_num):
                    for i1 in range(self.type_num):
                        for i2 in range(self.unit_num):
                            print('who-',i0,i1,i2)
                            for i in range(10):
                                print(arrayobs[i0,i1,i2,:,:,i])"""
            #print(self.arrayobs.shape)
            #print(board)
            if self.obs.step>=7: exit()
        return arrayobs,arrayrew,self.envH.done,self.info
    def getDirTo(self,fromPos, toPos, size):
        fromX, fromY = divmod(fromPos[0],size), divmod(fromPos[1],size)
        toX, toY = divmod(toPos[0],size), divmod(toPos[1],size)
        if fromY < toY: return ShipAction.NORTH
        if fromY > toY: return ShipAction.SOUTH
        if fromX < toX: return ShipAction.EAST
        if fromX > toX: return ShipAction.WEST
    def ruleagent(self,obs, config, playernum):
        size = config.size
        board = Board(obs, config)
        me = board.players[playernum]#board.current_player
        if len(me.ships) <= 0 and len(me.shipyards) > 0:
            me.shipyards[0].next_action = ShipyardAction.SPAWN
        if len(me.shipyards) == 0 and len(me.ships) > 0:
            me.ships[0].next_action = ShipAction.CONVERT
        if not self.rulemoveon: return me.next_actions ######
        for ship in me.ships:
            if ship.next_action == None:
                if ship.halite < 200: self.ship_states[playernum][ship.id] = "COLLECT"
                if ship.halite > 500: self.ship_states[playernum][ship.id] = "DEPOSIT"
                if self.ship_states[playernum][ship.id] == "COLLECT":
                    if ship.cell.halite < 100:
                        neighbors = [ship.cell.north.halite, ship.cell.east.halite, 
                                     ship.cell.south.halite, ship.cell.west.halite]
                        best = max(range(len(neighbors)), key=neighbors.__getitem__)
                        ship.next_action = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST][best]
                if self.ship_states[playernum][ship.id] == "DEPOSIT":
                    direction = self.getDirTo(ship.position, me.shipyards[0].position, size)
                    if direction: ship.next_action = direction
        return me.next_actions
    def render(self, mode='rgb_array', close=False):
        args = [self.envH.state, self.envH]
        out = self.envH.renderer(*args[:self.envH.renderer.__code__.co_argcount])
        self.real_screen.fill(BLACK)
        smallscreensize = self.boardsize*30+200
        fontsize, linespace, startline, vspace = 18, 18, 0, 0#smallscreensize-300, 0
        myfont = pygame.font.SysFont("monospace", fontsize)# font8=6px
        linestring = myfont.render('ORDER:%s' %str(self.player_order),True, BGRAY)
        self.player_index = np.argsort(np.array(self.player_order))
        self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
        startline+=linespace
        for iplayer,playerindex in enumerate(self.player_index):
            itype = 0
            linestring = myfont.render('ACTS %s:%s' %(str(iplayer),str(self.actions[playerindex][itype])),True, BGRAY)
            self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
            startline+=linespace#*self.play_num
        for iplayer,actions in enumerate(self.actions_string):
            linestring = myfont.render('ACTS %s:%s' %(str(iplayer),str(actions)),True, BGRAY)
            self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
            startline+=linespace#*self.play_num
        linestring = myfont.render('INIT:%s' %str(self.initall),True, BGRAY)
        self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
        startline+=linespace
        for iplayer,playerindex in enumerate(self.player_index):
            for itype,reward in enumerate(self.reward[playerindex]):
                linestring = myfont.render('REWA %s %s:%s %s' %(str(iplayer),str(itype),str(self.rewall[playerindex][itype]),str(reward)),True, BGRAY)
                self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
                startline+=linespace
        linestring = myfont.render('SHIP:%s' %str(self.shipbuild),True, BGRAY)
        self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
        startline+=linespace
        linestring = myfont.render('YARD:%s' %str(self.yardbuild),True, BGRAY)
        self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
        startline+=linespace
        linestring = myfont.render('STEP:%s' %str(self.envH.state[0].observation.step),True, BGRAY)
        self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
        startline+=linespace
        for iplayer,playernum in enumerate(self.player_order):
            self.cargo = 0
            for inum,shipid_shipvalue in enumerate(list(self.envH.state[0].observation.players[iplayer][SHIPS].items())):
                shipid, shipvalue = shipid_shipvalue
                self.cargo+=shipvalue[CARGO]
            linestring = myfont.render('%s HALS:%6s CARG:%6s' %(str(iplayer),str(self.envH.state[0].observation.players[iplayer][HALITE]),str(self.cargo)),True, BGRAY)
            self.real_screen.blit(linestring,(smallscreensize,linespace*0+startline))
            startline+=linespace
        lines = out.split('\n')
        for i,line in enumerate(lines):
            linestring = myfont.render('%s' %str(line),True, BGRAY)
            self.real_screen.blit(linestring,(0,linespace*i+smallscreensize))
        # self.channel_num, boardsize*boardsize
        #self.playermap = self.playermap.swapaxes(0,1).reshape(self.boardsize,self.boardsize,self.channel_num)
        arraymap = np.zeros([self.boardsize,self.boardsize,3])
        colorvector = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1]]
        if self.player_order[0]==0: player_order,k = [0,1,2,3],0
        if self.player_order[0]==1: player_order,k = [1,0,3,2],1
        if self.player_order[0]==2: player_order,k = [2,3,0,1],3
        if self.player_order[0]==3: player_order,k = [3,2,1,0],2
        for iplayer,playernum in enumerate(player_order):
            arraymap[:,:,0]+=(self.playermap[:,:,iplayer*self.layers_each+self.hold_layer]!=0)*255*colorvector[playernum][0]*0.3
            arraymap[:,:,1]+=(self.playermap[:,:,iplayer*self.layers_each+self.hold_layer]!=0)*255*colorvector[playernum][1]*0.3
            arraymap[:,:,2]+=(self.playermap[:,:,iplayer*self.layers_each+self.hold_layer]!=0)*255*colorvector[playernum][2]*0.3
            arraymap[:,:,0]+=(self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]!=0)*255*colorvector[playernum][0]*0.2
            arraymap[:,:,1]+=(self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]!=0)*255*colorvector[playernum][1]*0.2
            arraymap[:,:,2]+=(self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]!=0)*255*colorvector[playernum][2]*0.2
            arraymap[:,:,0]+=self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]*colorvector[playernum][0]*0.25
            arraymap[:,:,1]+=self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]*colorvector[playernum][1]*0.25
            arraymap[:,:,2]+=self.playermap[:,:,iplayer*self.layers_each+self.cargo_layer]*colorvector[playernum][2]*0.25
        arraymap[:,:,0]+=self.playermap[:,:,self.play_num*self.layers_each+self.halite_layer]*colorvector[self.play_num][0]*0.25
        arraymap[:,:,1]+=self.playermap[:,:,self.play_num*self.layers_each+self.halite_layer]*colorvector[self.play_num][1]*0.25
        arraymap[:,:,2]+=self.playermap[:,:,self.play_num*self.layers_each+self.halite_layer]*colorvector[self.play_num][2]*0.25
        arraymap = arraymap.swapaxes(0,1).astype(np.uint8)
        surf = pygame.surfarray.make_surface(arraymap)
        surf = pygame.transform.scale(surf, [smallscreensize,smallscreensize])
        self.real_screen.blit(surf,(0,0))
        #pygame.image.save(surf, 'arraymap.png')
        """printobs = np.around(arrayobs, decimals=1)
        fontsize, linespace, startline, vspace = 15, 15, 0, 100
        for i in range(printobs[0].shape[0]):
            for j in range(printobs[0].shape[1]):
                for k in range(printobs[0].shape[2]):
                    linestring = myfont.render('|%s|' %str(printobs[0][i][j][k]),True, BGRAY)
                    self.real_screen.blit(linestring,(smallscreensize+vspace*j,linespace*(i*3+k)+startline))
            startline+=linespace"""
        pygame.display.update() #pygame.display.flip()
        frame = pygame.surfarray.array3d(self.real_screen).swapaxes(0,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    def close(self):
        pass
    def seed(self, seed=None):
        random.seed(seed)
from pathlib import Path
root_path = Path(__file__).parent.resolve()
def myrenderer(state, env):
    config = env.configuration
    size = config.size
    obs = state[0].observation

    board = [[h, -1, -1, -1] for h in obs.halite]
    for index, player in enumerate(obs.players):
        _, shipyards, ships = player
        for shipyard_pos in shipyards.values():
            board[shipyard_pos][1] = index
        for ship in ships.values():
            ship_pos, ship_halite = ship
            board[ship_pos][2] = index
            board[ship_pos][3] = ship_halite

    col_divider = "|"
    row_divider = "+" + "+".join(["-------"] * size) + "+\n"

    out = row_divider
    for row in range(size):
        for col in range(size):
            _, _, ship, ship_halite = board[col + row * size]
            out += col_divider + (
                f"{min(int(ship_halite), 9999)}S{ship}" if ship > -1 else ""
            ).ljust(7)
        out += col_divider + "\n"
        for col in range(size):
            halite, shipyard, _, _ = board[col + row * size]
            if shipyard > -1:
                out += col_divider + f"SY{shipyard}".ljust(7)
            else:
                out += col_divider + str(min(int(halite), 9999)).rjust(7)
        out += col_divider + "\n" + row_divider

    return out
