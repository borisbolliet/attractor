def record(x, seen=set()):
    seen.add(x)
    return sorted(seen)
print(record(3), record(1), record(3))
def record2(x, seen=None):
    if seen is None:
        seen = set()
    seen.add(x)
    return sorted(seen)
print(record2(3), record2(1), record2(3))
