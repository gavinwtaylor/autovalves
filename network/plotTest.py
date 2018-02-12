import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

trainx=[1,2,3,4]
trainy=[100,50,20,10]
testx=[2,4]
testy=[60,15]

plt.plot(trainx,trainy,'r',testx,testy,'b')
plt.savefig('testfig.pdf')
