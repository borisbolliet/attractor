from dataclasses import dataclass, field
@dataclass
class Log:
    entries: list = field(default_factory=list)
class PlainLog:
    def __init__(self, entries=[]):
        self.entries = entries
r1, r2 = Log(), Log()
r1.entries.append("start")
print(r1.entries, r2.entries, r1.entries is r2.entries)
p1, p2 = PlainLog(), PlainLog()
p1.entries.append("start")
print(p1.entries, p2.entries, p1.entries is p2.entries)
try:
    @dataclass
    class Bad:
        entries: list = []
except ValueError as e:
    print("ValueError:", e)
