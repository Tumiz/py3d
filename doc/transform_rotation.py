from scenario import *
scenario=Scenario()
c=XYZ()
c.scale=Vector3(3,3,3)
c.line_width=2
p=XYZ()
p.position=c.transform_position(Vector3(0.5,0.5,0.5))
p.rotation=c.rotation
p.line_width=2
scenario.add(c,p)

scenario.render()