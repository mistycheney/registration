import base64
import sys
import cPickle as pickle
import subprocess 

sp = subprocess.Popen(['/bin/bash','-i','-c','hdfs -cat %s/part*'% sys.argv[1]], stdout=subprocess.PIPE)
output = sp.communicate()[0]

for line in output.split('\n'):
    if len(line) < 1: continue
    line_split = line.split('\t',1)
    key = line_split[0]
    content = line_split[1]
    scores = pickle.loads(base64.b64decode(content,'-_'))
 #   print scores
    pickle.dump(scores, open(sys.argv[2]+'_%s.p' % key, 'wb'))
