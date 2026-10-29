def announce(name):
    def deco(f):
        print(f"{name} applied")
        def wrapper(*args, **kwargs):
            print(f"{name} called")
            return f(*args, **kwargs)
        return wrapper
    return deco
@announce("outer")
@announce("middle")
@announce("inner")
def g():
    print("g runs")
print("--- one call")
g()
