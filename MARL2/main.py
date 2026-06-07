import optuna, joblib
import numpy as np
from collections import deque
from statistics import mean, stdev
import matplotlib.pyplot as plt
import cProfile, pstats
import random, os, time, copy
import argparse, json, pprint, tqdm
import envirs, agents, algos
from algos.optunasetting import optunasetting, optunascoring
def get_time():
    return np.array([time.process_time(),time.perf_counter()])
def trainloop(args):
    pprint.pprint(args)
    env,envinfo = envirs.getEnv(args)
    agent = agents.getAgent(env,envinfo,args)
    obs = env.reset()
    #test_args = copy.deepcopy(args)
    #test_args.env_num = 1
    #test_env,test_obs = envirs.getEnv(test_args)
    starttime = time.time()
    starttcpu = get_time()
    if args.timer:
        iterator_forward_time = 0
        step_forward_time, agent_update_time, other_in_iterator_time = 0, 0, 0
        agent_getaction_time, env_step_time, agent_memoexps_time = 0, 0, 0
        savepoint_time, optuna_stop_time = 0, 0
    last_scores = deque(maxlen=100)
    last_losses = deque(maxlen=100)
    if args.plotscore:
        last_scores_best = float("-inf")
        last_scores_means = []
        loss_means_v, loss_means_a, curve_steps = [], [], []
        live_curve_path = args.exp_dir+str(args.env_seed)+'_live_curve.png'
        live_csv_path   = args.exp_dir+str(args.env_seed)+'_live_curve.csv'
        def _save_live_curve():
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                ax1.plot(curve_steps, last_scores_means, color='tab:blue')
                ax1.set_ylabel('mean score (last 100 eps)'); ax1.grid(True, alpha=0.3)
                ax1.set_title(f'{args.env_name}  seed {args.env_seed}  step {curve_steps[-1] if curve_steps else 0}/{args.max_train_steps}')
                ax2.plot(curve_steps, loss_means_v, color='tab:red',   label='value loss')
                ax2.plot(curve_steps, loss_means_a, color='tab:orange',label='action loss')
                ax2.set_xlabel('update step'); ax2.set_ylabel('loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
                fig.tight_layout(); fig.savefig(live_curve_path, dpi=120); plt.close(fig)
                with open(live_csv_path, 'w') as fcsv:
                    print('step,score,value_loss,action_loss', file=fcsv)
                    for s, sc, lv, la in zip(curve_steps, last_scores_means, loss_means_v, loss_means_a):
                        print(f'{s},{sc},{lv},{la}', file=fcsv)
            except Exception as e:
                print('live curve save failed:', e)
    print(args.env_seed,':',args.fin_seed)
    if args.to_test:
        agent.load()
        for t in tqdm.tqdm(range(int(args.max_train_steps*args.testlength))):
            if args.render: env.render(mode='rgb_array')
            act, act_info = agent.getaction(obs,explore=False)
            new_obs, rew, done, info = env.step(act)
            agent.memoexps(new_obs, rew, done, info)
            obs = new_obs
    else:#synchronized multienv oneagent
        if args.to_load: agent.load()
        with open(args.exp_dir[:-1]+'_args', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        try:
            savepoint = list(np.array([i for i in range(1,args.savenum)])*args.max_train_steps//args.savenum)
            print(savepoint)
            iterator = tqdm.auto.tqdm(range(args.max_train_steps))
            for t in iterator:
                if args.timer: iterator_forward_start = get_time()
                for n in range(args.roll_num):
                    if args.timer: step_forward_start = get_time()
                    if args.timer: agent_getaction_start = get_time()
                    act, act_info = agent.getaction(obs,explore=True)
                    if args.timer: agent_getaction_time += get_time()-agent_getaction_start
                    if args.timer: env_step_start = get_time()
                    new_obs, rew, done, info = env.step(act)#must create a new_obs each step
                    if args.timer: env_step_time += get_time()-env_step_start
                    for infoi in info:
                        score_key = optunascoring() if args.optuna else 'last_score'
                        if score_key in infoi:
                            last_scores.append(infoi[score_key])
                    if args.plotscore:
                        if n%args.roll_num==0 and t%(max(1,args.max_train_steps//100))==0:
                            if len(last_scores)!=0:
                                last_scores_mean = mean(last_scores)
                                last_scores_means.append(last_scores_mean)
                                curve_steps.append(t)
                                loss_means_v.append(mean(x[0] for x in last_losses) if last_losses else 0.0)
                                loss_means_a.append(mean(x[1] for x in last_losses) if last_losses else 0.0)
                                _save_live_curve()
                    if n%args.roll_num==0 and t%(max(1,args.max_train_steps//200))==0:
                        elapsed = (time.time()-starttime)/60
                        score_str = f'{mean(last_scores):.1f}' if last_scores else 'n/a'
                        if last_losses:
                            vl = mean(x[0] for x in last_losses)
                            al = mean(x[1] for x in last_losses)
                            loss_str = f'v{vl:.3f}/a{al:.3f}'
                        else:
                            loss_str = 'n/a'
                        iterator.set_postfix(score=score_str, loss=loss_str, elapsed_min=f'{elapsed:.1f}')
                            #if last_scores_mean >= last_scores_best and t > args.max_train_steps*0.8:
                            #    last_scores_best = last_scores_mean
                            #    agent.save(str(args.env_seed))#+'_'+str(t))
                    if args.timer: agent_memoexps_start = get_time()
                    agent.memoexps(new_obs, rew, done, info)#must not to change new_obs
                    if args.timer: agent_memoexps_time += get_time()-agent_memoexps_start
                    obs = new_obs
                    if args.timer: step_forward_time += get_time()-step_forward_start
                if args.timer: agent_update_start = get_time()
                update_result = agent.update(t, args.max_train_steps, info_in={})
                if update_result is not None: last_losses.append(update_result[:2])  # (value_loss, action_loss)
                if args.timer: agent_update_time += get_time()-agent_update_start
                if args.timer: other_in_iterator_start = get_time()
                if args.timer: savepoint_start = get_time()
                if not args.optuna and t in savepoint: agent.save(str(args.env_seed)+'_'+str(t))
                if args.timer: savepoint_time += get_time()-savepoint_start
                if args.timer: optuna_stop_start = get_time()
                if args.optuna:
                    caltime = time.time()-starttime
                    caltcpu = get_time()-starttcpu
                    if caltcpu > args.timelim:
                        iterator.close()
                        break
                if args.timer: optuna_stop_time += get_time()-optuna_stop_start
                if args.timer: other_in_iterator_time += get_time()-other_in_iterator_start
                if args.timer: iterator_forward_time += get_time()-iterator_forward_start
            if not args.optuna: agent.save(str(args.env_seed)+'_'+str(t))#
            if args.plotscore:
                plt.figure()
                plt.plot(last_scores_means)
                plt.axvline(x=np.argmax(np.array(last_scores_means)))
                plt.savefig(args.exp_dir+str(args.env_seed)+'_curve.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
                plt.close()
        except KeyboardInterrupt:
            if not args.optuna: agent.save(str(args.env_seed)+'_'+str(t))
            if args.plotscore:
                plt.figure()
                plt.plot(last_scores_means)
                plt.axvline(x=np.argmax(np.array(last_scores_means)))
                plt.savefig(args.exp_dir+str(args.env_seed)+'_curve.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
                plt.close()
    try:    env.close()
    except: pass
    endtime = time.time()
    caltime = endtime-starttime
    endtcpu = get_time()
    caltcpu = endtcpu-starttcpu
    if args.timer:
        print('caltcpu:',np.round(caltcpu/60,2),' minutes')
        print('iterator_forward_time:',np.round(iterator_forward_time/60,2),' minutes')
        print('step_forward_time:',np.round(step_forward_time/60,2),' minutes')
        print('agent_update_time:',np.round(agent_update_time/60,2),' minutes')
        print('other_in_iterator_time:',np.round(other_in_iterator_time/60,2),' minutes')
        print('savepoint_time:',np.round(savepoint_time/60,2),' minutes')
        print('optuna_stop_time:',np.round(optuna_stop_time/60,2),' minutes')
        print('agent_getaction_time:',np.round(agent_getaction_time/60,2),' minutes')
        print('env_step_time:',np.round(env_step_time/60,2),' minutes')
        print('agent_memoexps_time:',np.round(agent_memoexps_time/60,2),' minutes')
    with open(args.exp_dir+'configs','a') as fconfigs:
        pprint.pprint(args,fconfigs)
        print(time.ctime(starttime),' ------ ',time.ctime(endtime),'        ',caltime//3600,'hours',np.round(caltime%3600/60,1),'minutes',file=fconfigs)
        print(starttcpu,' ------ ',endtcpu,'        ',caltcpu//3600,'hours',np.round(caltcpu%3600/60,1),'minutes',file=fconfigs)
    if args.optuna: args.optunascore = -last_scores_means[-1] #-mean(last_scores_means)/caltime
def mainloop(args):
    random.seed(args.env_seed)
    np.random.seed(args.env_seed)
    args.exp_dir = args.results_dir+'/'+args.env_name+'_'+str(args.env_num)#+'_'+str(args.roll_num)#+'_'+str(args.max_stepsM)
    envirs.add_strings(args)#,^=:
    agents.add_strings(args)
    algos.add_strings(args)
    args.exp_dir = args.exp_dir+'/'
    print('max stepsM: ', args.max_stepsM)
    args.max_steps = int(float(args.max_stepsM)*1e6)
    print('max steps: ',args.max_steps)
    print('lr_M: ', args.lr_M)
    args.lr = float(args.lr_M)*1e-6
    print('lr: ',args.lr)
    print('exp_dir length: ',len(args.exp_dir))
    args.checkpoint_dir, args.output_dir = args.exp_dir + 'checkpoints/', args.exp_dir + 'outputs/'
    for dir_ in [args.checkpoint_dir, args.output_dir]:
        if not os.path.exists(dir_): os.makedirs(dir_)
    args.rewardsname, args.lengthsname = args.exp_dir+'rewards_', args.exp_dir+'lengths_'
    args.max_train_steps = args.max_steps // args.roll_num // args.env_num
    if args.to_test:
        if args.render: args.env_num, args.rewardsname, args.lengthsname = 1, args.exp_dir+'run_rewards_', args.exp_dir+'run_lengths_'
        else:                         args.rewardsname, args.lengthsname = args.exp_dir+'test_rewards_', args.exp_dir+'test_lengths_'
    cProfile.runctx('trainloop(args)',   globals=globals(), locals={'args': args}, filename=args.exp_dir+str(args.env_seed)+'cProfile.out')
def optunaloop(args):
    try:
        lines = open('results/optunacurve','r').read().splitlines()
        elements = lines[0].split(',')
        scores = [float(element) for element in elements[:-1]]
    except:
        print('read results/optunacurve error, create new scores list...')
        scores = []
    def objective(trial):
        optunasetting(args,trial)
        args.plotscore = True
        optunascores = []
        for i in range(args.loopnum):
            args.env_seed = i
            mainloop(args)
            print('optunascore: ',args.optunascore)
            optunascores.append(args.optunascore)
        score = mean(optunascores)
        scores.append(score)
        return score
    try:
        study = joblib.load('results/study.pkl')
        print('Best trial until now:')
        print(' Value: ', study.best_trial.value)
        print(' Params: ')
        for key, value in study.best_trial.params.items():
            print(f'    {key}: {value}')
    except:
        study = optuna.create_study()
    try:
        study.optimize(objective, n_trials=args.optuna)
    except:
        print('error in optimize...')
        pass
    joblib.dump(study, 'results/study.pkl')
    with open('results/optunalogs','a') as foptunalogs:
        pprint.pprint(study.best_trial,foptunalogs)
        pprint.pprint(study.trials,foptunalogs)
    with open('results/optunacurve','w') as foptunacurve:
        for scorei in scores:
            print(scorei,end=',',file=foptunacurve)
    plt.figure()
    plt.plot(scores)
    plt.savefig('results/optunacurve.png', figsize=(16, 9), dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--timer', action='store_true', default=False, help='timer flag')
    parser.add_argument('--debug', action='store_true', default=False, help='debug flag')
    parser.add_argument('--optuna', type=int, default=0, help='optuna times')
    parser.add_argument('--loopnum', type=int, default=2, help='loop times')
    parser.add_argument('--timelim', type=int, default=300, help='train time limit')
    parser.add_argument('--plotscore', action='store_true', default=False, help='plotscore flag')
    parser.add_argument('--to-load', action='store_true', default=False, help='load previous agent flag')
    parser.add_argument('--to-test', action='store_true', default=False, help='testing flag')
    parser.add_argument('--testlength', type=float, default=1.0, help='test length (default: 1.0)')
    parser.add_argument('--render', action='store_true', default=False, help='render flag')
    parser.add_argument('--zoom-in', type=int, default=1, help='zoom-in size for render (default: 1)')
    parser.add_argument('--fps', type=int, default=30, help='fps for render (default: 30)')
    parser.add_argument('--width', type=int, default=600, help='width for render (default: 600)')
    parser.add_argument('--height', type=int, default=400, help='height for render (default: 400)')
    parser.add_argument('--results-dir', default='results', help='base directory for results (default: results)')
    parser.add_argument('--env-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--fin-seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gameflag', default='', help='atari or sc2 flag')
    parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4', help='environment to train on (default: BreakoutNoFrameskip-v4)')
    parser.add_argument('--start-step', type=int, default=0, help='number of environment step that start to train (default: 0)')
    parser.add_argument('--max-stepsM', default='10', help='number of environment steps to train (default: 10M)')
    parser.add_argument('--env-num', type=int, default=12, help='how many training CPU processes to use (default: 12)')
    parser.add_argument('--roll-num', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--savenum', type=int, default=5, help='number of saves')
    #parser.add_argument('--max-episodes', type=int, default=10e3, help='number of episodes to train (default: 10K)')
    parser.add_argument('--minmax-score', default='0.0,0.0', help='min max score (default: 0.0 to 0.0)')
    parser.add_argument('--min-unsolved', type=float, default=0.9, help='min unsolved ratio to stop training (default: 0.9)')
    parser.add_argument('--average-num', type=int, default=10, help='number of episodes to average (default: 10)')
    parser.add_argument('--solved-ratio', type=float, default=0.975, help='ratio of max score for solved (default: 0.975)')
    envirs.add_arguments(parser)
    agents.add_arguments(parser)
    algos.add_arguments(parser)
    args = parser.parse_args()
    if args.optuna: optunaloop(args)
    else:           mainloop(args)
    #cProfile.runctx('mainloop(args)',   globals=globals(), locals={'args': args}, filename=args.exp_dir+str(args.env_seed)+'cProfile.out')
#from absl import app
#from absl import flags
if __name__ == '__main__':
    #app.run(main)
    main()
"""
                if 1:
                    test_scores=[]
                    for i in range(10):
                        test_score = 0
                        while True:
                            test_act, test_act_info = agent.getaction(test_obs,explore=False)
                            test_new_obs, test_rew, test_done, test_info = test_env.step(test_act)
                            test_score += test_rew
                            test_obs = test_new_obs
                            if test_done: break
                        test_scores.append(test_score)
                    test_scores_mean = sum(test_scores) / len(test_scores) 
                    if test_scores_mean > test_scores_best:
                        test_scores_best = test_scores_mean
                        agent.save(str(args.env_seed)+'_'+str(t))
"""
