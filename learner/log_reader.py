import argparse
import os

if __name__ == '__main__':
  workdir=os.getenv("WORKDIR")
  parser=argparse.ArgumentParser()
  parser.add_argument('rank',help='rank of log file')
  parser.add_argument('jobnm', help='jobnumber of log file')
  args=parser.parse_args()
  
  with open(workdir+"/autovalves/learner/logs/log"+args.jobnm+"-rank"+args.rank+".txt") as f:
    for line in f:
      if "Number" in line and "Layers" in line:
        num = line.split()[-1]
        print(line, num)
      if "Width" in line:
        num=line.split()[-1]
        print(line, num) 
