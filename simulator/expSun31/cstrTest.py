import cstr

a=cstr.CSTREnv()
print(a.step((.5,20)))
a.reset()
print(a.step((.5,20)))
