import toweb
import math
s = toweb.Page("test")

s.plot(2, 3.3)
s.plot(4.7, 2.2)

for i in range(100):
    v = math.sin(i/10)
    s.plot(i, v)
    if v < 0.5:
        s.info("OK", i, v)
    elif v >= 0.5 and v <= 0.8:
        s.warn("Not good", i, v)
    else:
        s.err("Bad", i, v)
s.wait()
