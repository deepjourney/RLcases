import os,easydict,json,sys,cProfile, pstats

def print_pstats(filename,shownum):
    p = pstats.Stats(filename)
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(shownum)
    p.strip_dirs().sort_stats("tottime", "name").print_stats(shownum)

if __name__ == '__main__':
    suffixlist = str(sys.argv[1]).split(',')
    if len(sys.argv)>2 : shownum = int(sys.argv[2])
    else:                shownum = 30
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
                    print_pstats((rstfolder+file_)[:-5]+'/'+str(args.env_seed)+'cProfile.out',shownum)
                except:
                    print('WRONG!!!:',file_)
            else:
                if file_[-4:]!='args': continue
                args = easydict.EasyDict()
                with open(rstfolder+file_, 'r') as f:
                    args.__dict__ = json.load(f)
                print_pstats((rstfolder+file_)[:-5]+'/'+str(args.env_seed)+'cProfile.out',shownum)
