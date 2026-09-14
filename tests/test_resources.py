from __future__ import annotations

import pytest

from wnn.resources import DutyCycle, cpu_threads


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_duty_cycle_holds_busy_fraction():
    clock = FakeClock()
    duty = DutyCycle(0.8, clock=clock.time, sleep=clock.sleep)
    busy = 0.0
    for work in (0.4, 0.1, 1.2, 0.05):
        clock.now += work
        busy += work
        duty.tick()
    assert busy / clock.now == pytest.approx(0.8)
    assert clock.slept[0] == pytest.approx(0.1)


def test_duty_cycle_full_fraction_never_sleeps():
    clock = FakeClock()
    duty = DutyCycle(1.0, clock=clock.time, sleep=clock.sleep)
    clock.now += 3.0
    duty.tick()
    assert clock.slept == []


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_duty_cycle_rejects_bad_fraction(bad):
    with pytest.raises(ValueError):
        DutyCycle(bad)


def test_cpu_threads_respects_env(monkeypatch):
    monkeypatch.setenv("WNN_THREADS", "5")
    assert cpu_threads() == 5
    monkeypatch.delenv("WNN_THREADS")
    assert 1 <= cpu_threads() <= 16
