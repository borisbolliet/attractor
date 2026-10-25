fs = []
for k in (2, 3, 5):
    fs.append(lambda: k)
print(fs[0]())
del k
try:
    print(fs[0]())
except NameError as e:
    print("NameError:", e)
print(fs[0].__closure__)
def build():
    fs = []
    for k in (2, 3, 5):
        fs.append(lambda: k)
    print(fs[0]())
    del k
    try:
        print(fs[0]())
    except NameError as e:
        print("NameError:", e)
    return fs
fs = build()
