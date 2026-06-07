import myenv#, roboschool, pybullet_envs
def add_arguments(parser):
    parser.add_argument('--envir', default='full', help='envir to use: full | part | partdiv | skip (full)')
    parser.add_argument('--envparas', default='', help='myenv parameters')
    parser.add_argument('--npcparas', default='', help='units parameters')
    parser.add_argument('--agtparas', default='', help='units parameters')
    parser.add_argument('--envonoff', default='', help='env switches')
    parser.add_argument('--pobparas', default='', help='pomdp parameters')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.envir+'_'+args.envparas+'_'+args.npcparas+'_'+args.agtparas+'_'+args.envonoff#+'_'+args.pobparas
def getEnv(args):
    if args.envir=='full':
        from envirs.env_full import fEnv
    if args.envir=='part':
        from envirs.env_part import fEnv
    if args.envir=='skip':
        from envirs.env_skip import fEnv
    return fEnv(args)
