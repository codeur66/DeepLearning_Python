
# sd
from subprocess import check_output

import sys
import subprocess as sp
import tempfile
import os

## dsadsa
# asdsad
def git_hook():
    fd, path = tempfile.mkstemp()
    #dasda
    filename = sys.argv
    filename = str(filename[0])
    s1 = filename.split('/')[-1].split('.py')[0]

    with os.fdopen(fd, 'w') as tmp:
        with open(s1+'.py', 'r') as f:
            content = f.readlines()
            for line in content:
                if not line.startswith('#'):
                    line = line.rstrip()
                    tmp.write(line)

        sp.check_output(["git", "commit", ])



git_hook()