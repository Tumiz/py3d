from scenario import *
import time
from math import sin,cos

scenario=Scenario()

e=Line()
e.line_width=2
e.color=Color(r=1,b=1)

scenario.add(e)
t=0
while t<10:
    x=sin(5*t)*t
    y=cos(5*t)*t
    e.points.append([x,y,t])
    t+=0.01
    scenario.render()
    time.sleep(0.01)