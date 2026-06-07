from collections import OrderedDict
import numpy as np
from pprint import pprint
import json,sys,os,easydict,random
from matplotlib import pyplot as plt
from matplotlib import cm,axes
def fig_cscores(rstfoldername,expfoldername,rstfilename,args):
    lines = open(expfoldername+'/cscores','r').read().splitlines()
    print(len(lines))
    records = OrderedDict()
    for i,line in enumerate(lines):
        data = line.split(',')
        #print(data)
        thread = data[0]
        key = data[1]+','+data[2]
        value = np.array([int(data[3]),int(data[4]),int(data[5]),int(data[6])])
        if key in records: records[key] += value
        else:              records[key] = value
        #if i%100==0:
        #    for key,value in records.items():
        #        records[key] = value*0.5
    #print('records')
    #pprint(dict(records.items()))
    fullrecords = OrderedDict()
    for key,value in records.items():
        value0 = round(value[0]/value[3],2)
        value1 = round(value[1]/value[3],2)
        value2 = round(value[2]/value[3],2)
        value = [(value),np.array([value0,value1,value2])]
        fullrecords[key] = value
    print('fullrecords')
    pprint(dict(fullrecords.items()))
    sorted_fullrecords = sorted(fullrecords.items(), key=lambda x: x[1][0][0]/x[1][0][3], reverse=True)
    print('sorted_fullrecords')
    for i in range(len(sorted_fullrecords)):
        print(sorted_fullrecords[i])
    sorted_fullrecords = sorted(fullrecords.items(), key=lambda x: x[1][1][0], reverse=True)
    #print('sorted_fullrecords2')
    #for i in range(len(sorted_fullrecords)):
    #    print(sorted_fullrecords[i])

    agtrecords = {}
    npcrecords = {}
    for key,value in records.items():
        agtkey = key.split(',')[0]
        npckey = key.split(',')[1]
        if agtkey in agtrecords: agtrecords[agtkey]+=value.copy()
        else:                    agtrecords[agtkey]=value.copy()
        if npckey in npcrecords: npcrecords[npckey]+=value.copy()
        else:                    npcrecords[npckey]=value.copy()
        #pprint(dict(agtrecords.items()))
        #pprint(dict(npcrecords.items()))
    print('agt')
    fullagtrecords = OrderedDict()
    for key,value in agtrecords.items():
        value0 = round(value[0]/value[3],2)
        value1 = round(value[1]/value[3],2)
        value2 = round(value[2]/value[3],2)
        value = [(value),np.array([value0,value1,value2])]
        fullagtrecords[key] = value
    sorted_fullagtrecords = sorted(fullagtrecords.items(), key=lambda x: x[1][1][0], reverse=True)
    for i in range(len(sorted_fullagtrecords)):
        print(sorted_fullagtrecords[i])
    print('npc')
    fullnpcrecords = OrderedDict()
    for key,value in npcrecords.items():
        value0 = round(value[0]/value[3],2)
        value1 = round(value[1]/value[3],2)
        value2 = round(value[2]/value[3],2)
        value = [(value),np.array([value0,value1,value2])]
        fullnpcrecords[key] = value
    sorted_fullnpcrecords = sorted(fullnpcrecords.items(), key=lambda x: x[1][1][2], reverse=True)
    for i in range(len(sorted_fullnpcrecords)):
        print(sorted_fullnpcrecords[i])

    agtrecordsfair = {}
    npcrecordsfair = {}
    for key,value in fullrecords.items():
        agtkey = key.split(',')[0]
        npckey = key.split(',')[1]
        if agtkey in agtrecordsfair: agtrecordsfair[agtkey].append(value.copy()[1])
        else:                        agtrecordsfair[agtkey]=[value.copy()[1]]
        if npckey in npcrecordsfair: npcrecordsfair[npckey].append(value.copy()[1])
        else:                        npcrecordsfair[npckey]=[value.copy()[1]]
    print('agtfair')
    #pprint(dict(agtrecordsfair.items()))
    fullagtrecordsfair = OrderedDict()
    for key,value in agtrecordsfair.items():
        #weights = [pow(0.5,i) for i in range(len(value))].reverse()
        newvalue = np.round(np.average(value,axis=0),2)#,weights=weights),2)
        fullagtrecordsfair[key] = newvalue
    sorted_fullagtrecordsfair = sorted(fullagtrecordsfair.items(), key=lambda x: x[1][0], reverse=True)
    for i in range(len(sorted_fullagtrecordsfair)):
        print(sorted_fullagtrecordsfair[i])
    print('npcfair')
    #pprint(dict(npcrecordsfair.items()))
    fullnpcrecordsfair = OrderedDict()
    for key,value in npcrecordsfair.items():
        #weights = [pow(0.5,i) for i in range(len(value))].reverse()
        newvalue = np.round(np.average(value,axis=0),2)#,weights=weights),2)
        fullnpcrecordsfair[key] = newvalue
    sorted_fullnpcrecordsfair = sorted(fullnpcrecordsfair.items(), key=lambda x: x[1][2], reverse=True)
    for i in range(len(sorted_fullnpcrecordsfair)):
        print(sorted_fullnpcrecordsfair[i])

    xLabel = sorted([int(record[0]) for record in sorted_fullagtrecordsfair])
    yLabel = sorted([int(record[0]) for record in sorted_fullnpcrecordsfair], reverse=True)
    xLabel = [-1]+xLabel
    yLabel = yLabel+[-1]
    print(xLabel)
    print(yLabel)
    data = []
    for iy,y in enumerate(yLabel):
        tmp = []
        if str(y)=='-1':
            for ix,x in enumerate(xLabel):
                if str(x)=='-1':
                    dat = -1
                else:
                    dat = fullagtrecordsfair[str(x)][0]+fullagtrecordsfair[str(x)][1]/2
                dat = round(dat,3)
                tmp.append(dat)
        else:
            for ix,x in enumerate(xLabel):
                if str(x)=='-1':
                    dat = fullnpcrecordsfair[str(y)][0]+fullnpcrecordsfair[str(y)][1]/2
                else:
                    key = str(x)+','+str(y)
                    try:
                        dat = fullrecords[key][1][0]+fullrecords[key][1][1]/2
                    except:
                        dat = -1#2#-1#0.5#-0.1
                dat = round(dat,3)
                tmp.append(dat)
        data.append(tmp)
    print(data)
    fig= plt.figure(figsize=(20, 20), dpi=100, facecolor='azure')#"azure")
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel,size=18)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel,size=18)
    im = ax.imshow(data, cmap=plt.cm.brg, vmin=-1, vmax=1)#terrain)#gist_rainbow)#brg)#bwr)#_r)
    plt.colorbar(im)
    plt.show()
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
                    fig_cscores(rstfoldername,(rstfolder+file_)[:-5],rstfilename,args)#args.rewardsname)
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
                fig_cscores(rstfoldername,(rstfolder+file_)[:-5],rstfilename,args)#args.rewardsname)
