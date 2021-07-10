class P:
    x=1
class A(P):
    pass
class B(P):
    pass
print(P.x,A.x,B.x)
B.x=2
print(P.x,A.x,B.x)