from scenario import *
from time import sleep
c=Line()
c.points=[[0,0,0],[2,2,2]]
c.color=Color(r=1)
c.position=Vector3(0,0,0)
c.is_arrow=True
c.line_width=2
scen=Scenario()
scen.add(c)
while True:
    scen.render()
    sleep(1)