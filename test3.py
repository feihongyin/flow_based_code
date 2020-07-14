import numpy as np

a = np.array([[2, 3, 5], [4, 6, 7]])
b = np.array([[0, 1]])
print(a.shape)
print(b.shape)
c = b.T
print(c.shape)

d = np.concatenate((a, c), axis=1)
print(d.shape)
print("d:\n")
print(d)

print(d[:, :3].shape)
print(d[:, 3].shape)
print("***********")
np.random.shuffle(d)
print(d)



print("***********")

m = np.array([0, 1])
p = m.reshape(1, 2)
print(p.shape)
print(p)

mm = m.reshape((1, -1))
print(mm)
print(mm.shape)