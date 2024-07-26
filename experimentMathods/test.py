import numpy as np

u = day1 = np.array([10])

print(str(u.shape[0]))
v = np.asanyarray(u, dtype='float')
#print("1: " + str(u))
m = np.asarray(v[0],  order='c')
print("2: " + str(m)
      + " ====" + str(np.array([v[0]],  order='c').ndim)
      + " ====" + str(np.asarray([v[0]],  order='c').ndim))