import numpy as np
import argparse, json, easydict, random, os, time, pprint, cProfile, pstats, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import cycle
def fig_curves(args,filename):
    COLORS = cycle(['red','orange','green','cyan','blue','purple'])#,'black'])
    lines = [open(filename+str(i),"r").read().splitlines() for i in range(args.env_num)]
    plt.figure()
    regularxmeans, regularymeans = [], []
    for j in range(len(lines[0])):
        if len(lines[0][j])==0: continue#skip empty line
        color=next(COLORS)
        xytuples = []
        for i in range(args.env_num):
            records, avgnum = lines[i][j].split("|")[:-1], args.average_num#[:int(args.max_episodes)], args.average_num
            x, y = [], []
            for record in records:
                xe, ye = int(record.split(',')[0]), float(record.split(',')[1])
                x.append(xe)
                y.append(ye)
                xytuples.append((xe,ye))
            xmean = [x[k] for k in range(len(x))]#-avgnum+1)]
            ymean = [np.mean(y[max(0,l+1-avgnum):l+1]) for l in range(len(y))]#[np.mean(y[l:l+avgnum]) for l in range(len(y))]#-avgnum+1)]
            plt.plot(x,y,color=color,alpha=0.1,linewidth=0.1)
            plt.plot(xmean,ymean,color=color,alpha=0.3,linewidth=0.1)
        sortedxytuples = sorted(xytuples)
        sortedxytuplesx= [xytuple[0] for xytuple in sortedxytuples]
        sortedxytuplesy= [xytuple[1] for xytuple in sortedxytuples]
        #sortedxytuplesxmean = [sortedxytuplesx[k] for k in range(len(sortedxytuplesx)-avgnum+1)]
        #sortedxytuplesymean = [np.mean(sortedxytuplesy[l:l+avgnum]) for l in range(len(sortedxytuplesy)-avgnum+1)]
        #plt.plot(sortedxytuplesx,sortedxytuplesy,color=color,alpha=0.5,linewidth=0.1)######
        #plt.plot(sortedxytuplesxmean,sortedxytuplesymean,color=color,alpha=0.8,linewidth=0.3)
        regularx, regulary = [], []
        for istep in range(0,int(args.max_steps),int(args.max_steps/200)):######
            index = next((index for index,value in enumerate(sortedxytuplesx) if value>istep), len(sortedxytuplesx)-1)
            if index!=0:# continue
                regularx.append(istep)
                regulary.append(np.mean(sortedxytuplesy[max(0,index-100):index]))######
            else:
                regularx.append(istep)
                regulary.append(0.0)
        plt.plot(regularx,regulary,color=color,alpha=0.8,linewidth=0.3)######
        regularxmeans = regularx
        regularymeans.append(regulary)
    regularymeansmean= np.array(regularymeans).mean(axis=0)
    regularymeansvar = np.array(regularymeans).std(axis=0)
    plt.plot(regularxmeans,regularymeansmean,color='black',alpha=1.0,linewidth=0.3)
    plt.fill_between(regularxmeans, regularymeansmean-regularymeansvar, regularymeansmean+regularymeansvar,facecolor='black',alpha=0.5)
    max_steps = int(args.max_steps)#int(args.max_steps)*int(args.roll_num)*int(args.env_num)
    maxscore, minscore, n = args.max_score, -args.min_score, 10
    axes = plt.gca()
    axes.set_xticks(np.arange(0,max_steps,max_steps/n))
    #axes.set_yticks(np.arange(minscore,maxscore,(maxscore-minscore)/n))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlim([0,max_steps])
    #plt.ylim([minscore,maxscore])
    plt.tick_params(labelsize=8)
    plt.grid(linewidth=0.1)
    figname = args.exp_dir[:-1]+'_'+str(args.numparas)
    plt.savefig(figname+'_agtinfos'+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    plt.close()
    fall = open('res_'+args.env_name+'_agtinfos','a')
    print(figname,end='|',file=fall)
    for data in regularxmeans:
        print(data,end=',',file=fall)
    print('',end='|',file=fall)
    for data in regularymeansmean:
        print(data,end=',',file=fall)
    print('',end='|',file=fall)
    for data in regularymeansvar:
        print(data,end=',',file=fall)
    print('',file=fall)
    return
if __name__ == '__main__':
    folder = './results/'
    files =os.listdir(folder)
    files.sort()
    for file_ in files:
        try:
            if file_[-4:]!='args': continue
            args = easydict.EasyDict()
            with open(folder+file_, 'r') as f:
                args.__dict__ = json.load(f)
            fig_curves(args,args.agtinfosname)
        except:
            print('WRONG!!!:',file_)