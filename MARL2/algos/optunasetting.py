def optunascoring():
    return 'exinfos'#'last_score'
def optunasetting(args,trial):
    list_envparas = args.envparas.split('_')
    #args.envparas = list_envparas[0]+'_'+args.g_end+'_'+args.g_rew+'_'+list_envparas[3]+'_'+list_envparas[4]+'_'+list_envparas[5]
    #args.loopnum   = 2
    #args.timelim   = 600

    #5,5,3_13,987,200_1,-1,-10,100_0_28,1,1_3,4,10,1,1
    args.g_size = list_envparas[0]#'9,9,3,4'#'5,5,3,5'
    #g_end = trial.suggest_int('g_end',100,300)
    args.g_end  = list_envparas[1]#'10,1500'#+str(g_end*10)
    #print(args.g_end)

    lr_M  = trial.suggest_int('lr_M', 2, 100)
    args.lr_M   = lr_M*100
    #args.lr   = round(args.lr,2)
    print('optuna lr_M: ',args.lr_M)
    #args.lr   = trial.suggest_categorical('lr', [0.0004, 0.0007, 0.001, 0.0013, 0.0016])
    #return

    #g_rew.append(str(trial.suggest_categorical('rewrng',['0','1','-1'])))
    #g_rew.append(str(trial.suggest_categorical('rewrng',['0','1','-1'])))
    g_rews, rnum = [], 2
    g_rew = trial.suggest_int('timego', int(-1.00*pow(10,rnum)),  int(0.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('atkhit', int( 0.00*pow(10,rnum)),  int(1.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew2 = 0#trial.suggest_int('hitted', int( 0.00*pow(10,rnum)),  int(1.00*pow(10,rnum)))
    g_rew = -g_rew*g_rew2/pow(10,rnum)
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('draw',   int(-1.00*pow(10,rnum)),  int(0.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew = trial.suggest_int('lost',   int(-5.00*pow(10,rnum)), int(-1.00*pow(10,rnum)))
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum)))
    g_rew2 = int(50.00*pow(10,rnum))#trial.suggest_int('winn',   int(10.00*pow(10,rnum)), int(50.00*pow(10,rnum)))
    g_rew = -g_rew*g_rew2/pow(10,rnum)
    g_rews.append(str(round(g_rew/pow(10,rnum),rnum-1)))
    args.g_rew  = ','.join(g_rews)
    #args.g_rew  = '0,1,-1,0,-2,10'
    print(args.g_rew)
    #args.g_envp = '0'
    #args.g_npcp = '1,4,35,1,1'
    #args.g_agtp = '3,4,10,1,1'
    args.envparas = args.g_size+'_'+args.g_end+'_'+args.g_rew#+'_'+args.g_envp+'_'+args.g_npcp+'_'+args.g_agtp
    return




    #args.stack_num = trial.suggest_int('stack_num', 2, 4)
    args.res  = trial.suggest_categorical('res',['3,3,3,2,2,1,1,1,1,64,2^64,1,1^128,1,2',
                                                 '3,3,3,2,2,1,2,2,1,64,2^64,1,1^128,1,2'])
    args.cnn  = trial.suggest_categorical('cnn',['3,3,3,2,2,1,0,0,1,128,1^2,2,3,1,1,1,0,0,1,256,1',
                                                 '3,3,3,2,2,1,1,1,1,128,1^2,2,3,1,1,1,1,1,1,256,1',
                                                 '3,3,3,2,2,2,1,1,1,128,1^2,2,3,1,1,1,1,1,1,256,1'])
    args.mlp  = trial.suggest_categorical('mlp',['512','256','128'])
    #args.memoplace = trial.suggest_categorical('memoplace', ['agtcpu', 'algocpu', 'algogpu'])
    if args.aprxfunc == 'cnn3d': args.apfparas = args.cnn+'='+args.mlp
    if args.aprxfunc == 'res3d': args.apfparas = args.res+'='+args.mlp
    return

