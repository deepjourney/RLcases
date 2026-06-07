import numpy as np
import argparse, json, easydict, random, os, time, pprint, cProfile, pstats, sys, hashlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import cycle
robosnames = ['RoboschoolAnt-v1','RoboschoolHopper-v1','RoboschoolWalker2d-v1','RoboschoolHalfCheetah-v1','RoboschoolReacher-v1']
atarinames = ["BeamRiderNoFrameskip-v4","BreakoutNoFrameskip-v4","PongNoFrameskip-v4","QbertNoFrameskip-v4"]
COLORLIST  = ['red','orange','green','cyan','blue','purple']#,'black']
MARKERLIST = ['+', '.', 'o', '*']#",",".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","d","D","|","_","","","","",""
def fig_all_curves(filename, t_steps, fileheadname, filelistname, fileid):
    COLORS = cycle(COLORLIST)
    MARKERS= cycle(MARKERLIST)
    print(fileheadname)
    for i in range(fileid%len(COLORLIST)):
        print(i)
        color=next(COLORS)
    for i in range(abs(hash(fileheadname))%len(MARKERLIST)):
        marker=next(MARKERS)
    lines = open(filename,'r').read().splitlines()
    for i,line in enumerate(lines):
        elements = line.split('|')

        #results/:
        #drone-v0_10000000_64_5:
        #full_23_13,987_0,1,0,0_0_11_1_4_70,5,10^70,5,10=1:
        #imagine_4_5_none,none,0.0_agtcpu:
        #PTa2c1_0.001_const_0.1,5556,0.8_Adam:
        #one_0.99_0.5_0.01_0.5:
        #cnnmlp_7,7,2,2,128,1^3,3,1,1,256,1^7,7=512=64^64
        heads = elements[0].replace('/',':').split(':')
        print(heads)
        title = heads[1].split('_')[0]#.split('-')[0]
        env   = heads[2].split('_')
        agent = heads[3].split('_')
        method= heads[4].split('_')
        model = heads[5].split('_')
        apf   = ''#heads[6].split('_')
        #labels= ':'.join(heads[2:])
        if agent[0] == 'imagine' or agent[0]=='img':
            agentstring = 'independent learner'
            start_step  = 0
        elif agent[0] == 'reconimg':
            agentstring = 'incremental learner'
            start_step  = 300000
        else:
            agentstring = ''
            start_step  = 0
        labels= agentstring+':'+env[0]+' env'+'('+method[0][2:-1]+')' ######
        labels= apf#env[-2]+':'+method[1]+':'+method[-1]

        totalsteps = t_steps*10000
        minscore,maxscore = -50,50
        if title == 'RoboschoolAnt-v1':         minscore,maxscore = -0,3000
        if title == 'RoboschoolHalfCheetah-v1': minscore,maxscore = -0,3500
        if title == 'RoboschoolHopper-v1':      minscore,maxscore = -0,2500
        if title == 'RoboschoolReacher-v1':     minscore,maxscore = -50,25
        if title == 'RoboschoolWalker2d-v1':    minscore,maxscore = -0,1500

        if title == 'BeamRiderNoFrameskip-v4':  minscore,maxscore = -0,8000
        if title == 'BreakoutNoFrameskip-v4':   minscore,maxscore = -0,600#1000
        if title == 'PongNoFrameskip-v4':       minscore,maxscore = -25,25
        if title == 'QbertNoFrameskip-v4':      minscore,maxscore = -0,20000

        if title == 'LunarLander-v2':  minscore,maxscore = -200,300
        if title == 'LunarLanderContinuous-v2':   minscore,maxscore = -200,300
        if title == 'BipedalWalker-v2':       minscore,maxscore = -200,300
        if title == 'BipedalWalkerHardcore-v2':      minscore,maxscore = -200,300

        if title == 'MountainCar-v0':  minscore,maxscore = -250,-100
        if title == 'MountainCarContinuous-v0':   minscore,maxscore = -100,100
        if title == 'Pendulum-v0':       minscore,maxscore = -2100,-100
        if title == 'CartPole-v1':      minscore,maxscore = -100,600
        if title == 'Acrobot-v1':      minscore,maxscore = -700,0

        xs, ys, yv = [],[],[]
        for element in elements[1].split(',')[:-1]:
            xs.append(int(element)+start_step)
        for element in elements[2].split(',')[:-1]:
            ys.append(float(element))
        for element in elements[3].split(',')[:-1]:
            yv.append(float(element))
        xs = np.array(xs)
        ys = np.array(ys)
        yv = np.array(yv)

        plt.figure(title)
        color=next(COLORS)
        marker=next(MARKERS)
        markers_on=[-1]
        start = 3
        plt.plot(xs[start:],ys[start:],color=color,alpha=1.0,linewidth=0.5,marker=marker,markersize=3,markevery=markers_on,label=labels)
        plt.fill_between(xs[start:], ys[start:]-yv[start:], ys[start:]+yv[start:],facecolor=color,alpha=0.3)
        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0)
        plt.legend(loc='lower left') ######
        axes = plt.gca()
        axes.set_xticks(np.arange(0,totalsteps,totalsteps/10))
        axes.set_yticks(np.arange(minscore,maxscore,(maxscore-minscore)/10))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlim([0,totalsteps])
        plt.ylim([minscore,maxscore])
        plt.tick_params(labelsize=8)
        plt.title(title)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Episodic Reward')
        plt.savefig('res_'+filelistname+'_'+title+'_'+str(round(totalsteps/1000000,1))+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
if __name__ == '__main__':
    if len(sys.argv)>1 : suffixlist = str(sys.argv[1]).split(',')
    else:                suffixlist = ['']
    if len(sys.argv)>2 : curvelength = int(sys.argv[2])
    else:                curvelength = 5000
    files =os.listdir('./')
    files.sort()
    for ifile,file_ in enumerate(files):
        if os.path.isdir(file_): continue
        if file_[:7]!='results' or file_[-4:]=='.png': continue
        filehead = file_.split('_')[0][7:]
        if filehead in suffixlist:
            fileheadid = suffixlist.index(filehead)
            fig_all_curves(file_,curvelength,filehead,str(sys.argv[1]),fileheadid)
