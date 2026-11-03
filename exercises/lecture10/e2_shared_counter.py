CALLS = [0]
def counted(f):
    def wrapper(*args, **kwargs):
        CALLS[0] += 1
        return f(*args, **kwargs)
    wrapper.calls = CALLS
    return wrapper
@counted
def ping(): pass
@counted
def pong(): pass
ping(); pong(); pong()
print(ping.calls, pong.calls, ping.calls is pong.calls)
