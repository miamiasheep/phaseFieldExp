import os

FILENAME = 'expTestD'
Ds = [1, 2.25, 10]
for D in Ds:
    os.system('python3 allenChan.py --D {} --iteration 500 --interval 10'.format(D))

