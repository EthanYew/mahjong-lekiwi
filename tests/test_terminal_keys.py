"""Keys from the bench's own terminal (mahjong/terminal_keys.py): arrows and Esc as lerobot's flags, letters to
the bench, nothing while stopped, and the terminal put back as it was."""
import os
import pty
import time

from mahjong.terminal_keys import TerminalKeys, new_events, parse


def test_arrows_esc_and_letters_are_told_apart():
    assert parse(b"b\x1b[Cp") == ["b", "right", "p"]
    assert parse(b"\x1bOD") == ["left"]                           # application-mode arrows
    assert parse(b"\x1b[A\x1b[1;5Cd") == ["d"]                   # up, ctrl-right: not ours
    assert parse(b"\x1b", more=lambda: b"") == ["esc"]            # a lone ESC
    assert parse(b"\x1b", more=lambda: b"[C") == ["right"]        # an arrow split across reads


def wait_for(condition, timeout=2.0):
    end = time.time() + timeout
    while not condition() and time.time() < end:
        time.sleep(0.01)
    return condition()


def test_only_keys_typed_into_this_terminal_while_started_count():
    import termios
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "r")
    events, chars = new_events(), []
    keys = TerminalKeys(events, on_char=chars.append, stream=stream)
    assert keys.available
    before = termios.tcgetattr(slave)
    os.write(master, b"x")                                       # typed before the arm moved: flushed
    keys.start()
    os.write(master, b"b\x1b[C")
    assert wait_for(lambda: chars == ["b"] and events["exit_early"])
    assert not events["stop_recording"]
    os.write(master, b"\x1b")
    assert wait_for(lambda: events["stop_recording"])
    keys.stop()
    assert termios.tcgetattr(slave) == before                     # line input again for prompts
    os.write(master, b"p\n")
    time.sleep(0.1)
    assert chars == ["b"]                                         # stopped: not read
    keys.stop()                                                   # idempotent
    stream.close()
    os.close(master)


def test_without_a_terminal_it_does_nothing(tmp_path):
    with open(tmp_path / "f", "w") as f:
        keys = TerminalKeys(new_events(), stream=f)
        assert not keys.available
        keys.start()
        keys.stop()
