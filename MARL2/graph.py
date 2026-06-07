import numpy as np
import argparse, json, easydict, random, os, time, pprint, cProfile, pstats, sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import cycle
def fig_curves(rstfoldername,expfoldername,rstfilename,args):
    print('rstfoldername',rstfoldername)
    print('expfoldername',expfoldername)
    print('rstfilename',rstfilename)
    COLORS = cycle(['red','orange','green','cyan','blue','purple'])#,'black'])
    filename = expfoldername+'/'+rstfilename
    lines = [[line for line in open(filename+str(i),"r").read().splitlines() if len(line)!=0] for i in range(args.env_num)]
    #print(len(lines))
    #print(len(lines[0]))
    #for i in range(args.env_num):
    #    print(i,len(lines[i]))
    plt.figure()
    regularxmeans, regularymeans = [], []
    for j in range(len(lines[0])):
        if j%int(args.skip[1])==int(args.skip[0]): continue
        if len(lines[0][j])==0: continue#skip empty line
        color=next(COLORS)
        xytuples = []
        for i in range(args.env_num):
            try:
                records, avgnum = lines[i][j].split("|")[:-1], args.average_num#[:int(args.max_episodes)], args.average_num
            except:
                print(i,j)
                exit()
            x, y = [], []
            for record in records:
                xe, ye = int(record.split(',')[0]), float(record.split(',')[1])
                x.append(xe)
                y.append(ye)
                xytuples.append((xe,ye))
            xmean = [x[k] for k in range(len(x))]#-avgnum+1)]
            ymean = [np.mean(y[max(0,l+1-avgnum):l+1]) for l in range(len(y))]#[np.mean(y[l:l+avgnum]) for l in range(len(y))]#-avgnum+1)]
            plt.plot(x,y,color=color,alpha=0.1,linewidth=0.1)
            #print(rstfilename,i)
            #print('xmean',xmean)
            #print('ymean',ymean)
            plt.plot(xmean,ymean,color=color,alpha=0.3,linewidth=0.1)
        if rstfilename[:4]=='test': print(xytuples)
        sortedxytuples = sorted(xytuples)
        sortedxytuplesx= [xytuple[0] for xytuple in sortedxytuples]
        sortedxytuplesy= [xytuple[1] for xytuple in sortedxytuples]
        if rstfilename[:4]=='test': print(np.array(sortedxytuplesy).mean())
        #sortedxytuplesxmean = [sortedxytuplesx[k] for k in range(len(sortedxytuplesx)-avgnum+1)]
        #sortedxytuplesymean = [np.mean(sortedxytuplesy[l:l+avgnum]) for l in range(len(sortedxytuplesy)-avgnum+1)]
        #plt.plot(sortedxytuplesx,sortedxytuplesy,color=color,alpha=0.5,linewidth=0.1)######
        #plt.plot(sortedxytuplesxmean,sortedxytuplesymean,color=color,alpha=0.8,linewidth=0.3)
        regularx, regulary = [], []
        for istep in range(int(args.start_step),int(args.start_step+args.max_steps),int(args.max_steps/200)):######
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
    max_steps = int(args.max_steps+args.start_step)#int(args.max_steps)*int(args.roll_num)*int(args.env_num)
    maxscore, minscore, n = float(args.minmax_score.split(',')[1]), float(args.minmax_score.split(',')[0]), 10
    axes = plt.gca()
    axes.set_xticks(np.arange(0,max_steps,max_steps/n))
    plt.xlim([0,max_steps])
    #print(maxscore,minscore)
    if maxscore>0 or minscore<0:
        #print('setting ylim')
        diffscore = (maxscore - minscore)/n
        axes.set_yticks(np.arange(minscore-diffscore,maxscore+diffscore,diffscore))
        plt.ylim([minscore-diffscore,maxscore+diffscore])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tick_params(labelsize=8)
    plt.grid(linewidth=0.1)
    figname = expfoldername+rstfilename[5:7]+rstfilename[8:10]#+str(args.numparas)#args.exp_dir[7:-1]+'_'+str(args.numparas)
    figname = figname.replace('full','f').replace('none','n').replace('prev','p').replace('imagine','img')
    print('figname',figname)
    print('figname[:252]',figname[:252])
    plt.savefig(figname[:252]+(args.skip[0])+(args.skip[1])+'.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    plt.close()
    fall = open(rstfoldername+'_'+args.env_name,'a')
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
    suffixlist = str(sys.argv[1]).split(',')
    if len(sys.argv)>2 : rstfilename = str(sys.argv[2])
    else:                rstfilename = 'rewards_'
    for suffix in suffixlist:
        rstfoldername = 'results'+suffix
        rstfolder = rstfoldername+'/'
        files =os.listdir(rstfolder)
        files.sort()
        for file_ in files:
            if 0:
                try:
                    if file_[-4:]!='args': continue
                    args = easydict.EasyDict()
                    with open(rstfolder+file_, 'r') as f:
                        args.__dict__ = json.load(f)
                    if len(sys.argv)>3 : args.minmax_score = str(sys.argv[3])
                    if len(sys.argv)>4 : args.skip = str(sys.argv[4])
                    else:                args.skip = '1,1'
                    fig_curves(rstfoldername,(rstfolder+file_)[:-5],rstfilename,args)#args.rewardsname)
                except:
                    print('WRONG!!!:',file_)
            else:
                if file_[-4:]!='args': continue
                args = easydict.EasyDict()
                with open(rstfolder+file_, 'r') as f:
                    args.__dict__ = json.load(f)
                if len(sys.argv)>3 : args.minmax_score = str(sys.argv[3])
                if len(sys.argv)>4 : args.skip = str(sys.argv[4]).split(',')
                else:                args.skip = ['1','1']
                fig_curves(rstfoldername,(rstfolder+file_)[:-5],rstfilename,args)#args.rewardsname)
