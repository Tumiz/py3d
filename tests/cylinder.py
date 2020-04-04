from scenario import *
c=Cylinder()
c.top_radius=0
c.bottom_radius=0.1
c.height=0.3
c.color=Color(r=1)
c.position=Vector3(1,0,0)
scen=Scenario()
scen.add(c)
scen.render()