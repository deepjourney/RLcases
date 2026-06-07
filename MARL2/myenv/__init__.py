from gymnasium.envs.registration import register
register(id='shipS-v0',     entry_point='myenv.shipS:ShipS')
register(id='drone-v0',     entry_point='myenv.drone:Drone')
register(id='escape-v0',    entry_point='myenv.escape:Escape')
register(id='microa-v0',    entry_point='myenv.microa:Micro')
register(id='scan-v0',      entry_point='myenv.scan:Scan')
register(id='microaM-v0',   entry_point='myenv.microaM:MicroM')
register(id='scanM-v0',     entry_point='myenv.scanM:ScanM')
register(id='microaMs-v0',  entry_point='myenv.microaMs:MicroMs')
register(id='microaMsM-v0', entry_point='myenv.microaMsM:MicroMsM')
register(id='microMM-v0',   entry_point='myenv.microMM:MicroMM')
register(id='scanMM-v0',    entry_point='myenv.scanMM:ScanMM')
register(id='csc2-v0', entry_point='myenv.csc2:Csc2')
register(id='haliteM-v0', entry_point='myenv.haliteM:HaliteM')
# pygame 相关的渲染/精灵工具已移到 myenv/_render.py，自定义环境改从那里导入；
# 这样 Atari 等训练导入 myenv 时不会加载 pygame，避免与 opencv 的 SDL 重复加载警告。
