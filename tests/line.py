from scenario import *
import time
from math import sin,cos

scenario=Scenario()

e=Line()
e.line_width=2
e.color=Color(r=1,b=1)
e.is_arrow=True
e.arrow_length=0.6
e.arrow_radius=0.1
scenario.add(e)
t=0
while True:
    x=sin(5*t)*t
    y=cos(5*t)*t
    e.points.append([x,y,t])
    print(len(e.points))
    t+=0.01
    scenario.render()
    time.sleep(0.01)