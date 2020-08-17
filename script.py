import os


def run(cmd):
    print(cmd)
    os.system(cmd)

FILENAME = 'expTestD'
Ds = [1, 2.25, 10]
for D in Ds:
    run('python3 allenChan.py --filename --D {} --iteration 500 --interval 10'.format(FILENAME, D))

