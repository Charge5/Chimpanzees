log_path = '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/log-results3.txt'

f = open(log_path)
for i in f.readlines():
    if i.startswith(('Time for iteration', 'Iteration')):
        print(i)