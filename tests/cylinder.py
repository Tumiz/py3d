from scenario import *
from math import pi
c=Cylinder()
c.top_radius=1
c.bottom_radius=2
c.height=1
c.color=Color(r=1)
# c.position=Vector3(1,0,0)
# c.rotation=Vector3(0,pi/2,pi/2)
scen=Scenario()
scen.add(c)
scen.render()