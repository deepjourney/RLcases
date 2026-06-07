import gym, myenv, pygame, cProfile, pstats, json, easydict
def main():
    pygame.init()
    with open('./myenv/envinfo.json', 'r') as envinfo_file:
        envinfo_args_dict = json.load(envinfo_file)
    args = easydict.EasyDict(envinfo_args_dict)
    env = gym.make('microMM-v0')#args.env_name)
    env.setsize(32,10)#args.times*2,args.timelag/1) ###
    env.reset()
    quit=False
    move = {pygame.K_LEFT:0, pygame.K_RIGHT:0, pygame.K_UP:0, pygame.K_DOWN:0}
    step,target = 0,0
    while True:
        env.render("human")
        for event in pygame.event.get():
            if event.type == pygame.QUIT: quit=True
            if event.type == pygame.KEYDOWN:
                if event.key in move: move[event.key] = 1
            elif event.type == pygame.KEYUP:
                if event.key in move: move[event.key] = 0
            if event.type == pygame.KEYDOWN:
                print(pygame.key.name(event.key))
                if pygame.key.name(event.key)=='c': target+=1
        if 0:#num_act=8
            action=0
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=0
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=1
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=2
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=3
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=8
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=4
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=5
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=6
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=7
        else:
            action=8
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=0
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=0
            if move[pygame.K_LEFT] > move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=0
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=2
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=4
            if move[pygame.K_LEFT] ==move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=3
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   > move[pygame.K_DOWN]: action=1
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   ==move[pygame.K_DOWN]: action=1
            if move[pygame.K_LEFT] < move[pygame.K_RIGHT] and move[pygame.K_UP]   < move[pygame.K_DOWN]: action=1
        act = []
        for i in range(7):
            if i==target%7:
                act.append(action)
            else:
                act.append(4)
        if step%600==0: obs, rew, done, info = env.step(act)
        if done==True: env.reset()
        if quit==True: break
    env.close()

if __name__ == '__main__':
    cProfile.run('main()', filename='cProfile.out')
    p = pstats.Stats("cProfile.out")
    #p.strip_dirs().sort_stats(-1).print_stats()
    #p.strip_dirs().sort_stats("name").print_stats(0.3)
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(30)
    p.strip_dirs().sort_stats("tottime", "name").print_stats(30)
    #p.print_callers(0.5, "sum_num")
    #p.print_callees("test")
