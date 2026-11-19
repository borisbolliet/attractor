"""Tests of the timing helper."""
from attractor.tools import counted, median_time


def test_median_time_makes_one_untimed_call_then_repeat_calls():
    f = counted(lambda x: x + 1)
    seconds = median_time(f, 1, repeat=5)
    assert f.calls == 6
    assert seconds >= 0.0
