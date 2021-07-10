import toweb
import math

s = toweb.Chart("test")
s.plot(2, 3.3)
s.plot(4.7, 2.2)
log=toweb.Log()

for i in range(100):
    v = math.sin(i/10)
    s.plot(i, v)
    if v < 0.5:
        log.info("OK", i, v)
    elif v >= 0.5 and v <= 0.8:
        log.warn("Not good", i, v)
    else:
        log.err("Bad", i, v)
s.wait()
