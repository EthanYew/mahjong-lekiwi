#!/usr/bin/env python
"""Watch the wrist camera's USB link while you try things, to find which part fails. Moves nothing.

  python scripts/wrist_usb_watch.py --label still --seconds 600     arm left alone: the control
  python scripts/wrist_usb_watch.py --label by-hand --seconds 300   motors off, move the arm by hand: drops mean the cable
  python scripts/wrist_usb_watch.py --kernel-only --label bench     in a 2nd terminal during a bench or teleop run, which
                                                                    holds the camera: motors on, the same moves
  python scripts/wrist_usb_watch.py --report                        every watch so far, and how real losses ended

Swap one part at a time (cable, then port) and give each a label. It reports a drop and waits for the camera; it never
tries to recover it (no power cycle), since that would hide what the link does by itself.
"""
import argparse
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mahjong import config, session, wrist_usb    # noqa: E402

GAP_S = 0.15        # a frame this late is worth a line; 30fps is 0.033s
KINDS = (("emi", "EMI?"), ("disconnect", "USB disconnect"), ("protocol_error", "-71"), ("power_cycle", "attempt power cycle"),
         ("no_enumerate", "unable to enumerate"), ("reset", "reset high-speed"), ("connect", "New USB device found"))


def classify(line, port):
    """What a kernel log line says about `port`, or None if it isn't about it."""
    hub, _, number = port.rpartition(".")
    if hub:
        name = f"{hub}-port{number}"
    else:
        bus, _, number = port.partition("-")
        name = f"usb{bus}-port{number}"
    if not re.search(rf"(usb|uvcvideo) {re.escape(port)}[: ]|\b{re.escape(name)}\b", line):
        return None
    return next((kind for kind, text in KINDS if text in line), "other")


class KernelLog:
    """New kernel log lines, as they arrive."""

    def __init__(self):
        self.process = subprocess.Popen(["journalctl", "-k", "-f", "-n", "0", "--no-pager", "-o", "short-iso-precise"],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())

    def drain(self):
        lines = []
        while not self.lines.empty():
            lines.append(self.lines.get())
        return lines

    def close(self):
        self.process.terminate()


class SimKernelLog:
    """What a drop looks like, once."""

    def __init__(self, port):
        self.lines = [f"kernel: usb {port}: USB disconnect, device number 5", f"kernel: usb {port}: New USB device found"]

    def drain(self):
        lines, self.lines = self.lines, []
        return lines

    def close(self):
        pass


def watch(kernel, port, seconds, log, open_camera=None, clock=time.monotonic, sleep=time.sleep, lost_s=None):
    """Log frames and kernel lines for `seconds`. open_camera=None watches only the kernel. Returns the totals."""
    from mahjong.camera import LOST_S
    lost_s = LOST_S if lost_s is None else lost_s
    totals = {"seconds": 0.0, "frames": 0, "drops": 0, "worst_gap_s": 0.0, "back_after_s": [],
              **{kind: 0 for kind, _ in KINDS}, "other": 0}
    start = clock()

    def read_kernel():
        for line in kernel.drain():
            kind = classify(line, port)
            if kind:
                totals[kind] += 1
                log("kernel", kind=kind, line=line)

    camera = None
    try:
        if open_camera:
            try:
                camera = open_camera()
            except Exception as e:     # missing at the start: reported as a drop, like any other
                log("not_open", error=str(e))
        last = clock()
        while clock() - start < seconds:
            read_kernel()
            if camera is None and not open_camera:
                sleep(0.2)
                continue
            ok = camera.read()[0] if camera is not None else False
            now = clock()
            if ok:
                gap = now - last
                if totals["frames"] and gap > GAP_S:
                    totals["worst_gap_s"] = max(totals["worst_gap_s"], round(gap, 3))
                    log("gap", seconds=round(gap, 3))
                totals["frames"] += 1
                last = now
            elif now - last >= lost_s:
                totals["drops"] += 1
                log("lost", frames_so_far=totals["frames"])
                if camera is not None:
                    camera.release()
                camera, lost_at = None, now
                while camera is None and clock() - start < seconds:
                    read_kernel()
                    try:
                        camera = open_camera()
                    except Exception:
                        sleep(0.1)
                if camera is not None:
                    totals["back_after_s"].append(round(clock() - lost_at, 2))
                    log("back", seconds=totals["back_after_s"][-1])
                last = clock()
            else:
                sleep(0.01)
    except KeyboardInterrupt:
        pass
    read_kernel()
    if camera is not None:
        camera.release()
    totals["seconds"] = round(clock() - start, 1)
    return totals


def summary(label, totals):
    hours = max(totals["seconds"], 1e-9) / 3600
    return (f"{label:<16} {totals['seconds'] / 60:>6.1f} min  {totals['drops']:>3} drops  {totals['disconnect']:>3} disconnects "
            f"({totals['disconnect'] / hours:>5.1f}/h)  {totals['emi']:>2} EMI?  {totals['protocol_error']:>2} -71  "
            f"worst gap {totals['worst_gap_s']:.2f}s")


def report(sessions=None):
    folders = sorted((sessions or config.SESSIONS).glob("*"))
    all_events = [(folder, session.read_events(folder)) for folder in folders]
    print("Watches:")
    for folder, events in all_events:
        end = next((e for e in events if e["type"] == "watch_end"), None)
        if end:
            print(f"  {folder.name[:15]} {summary(end['label'], end['totals'])}")
    print("\nHow real wrist camera losses ended, over every session:")
    print("  " + (wrist_usb.recovery_summary([e for _, events in all_events for e in events]) or "none logged yet"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--label", default="watch", help="what you're trying, to compare runs by: still, by-hand, new-cable, port-7")
    p.add_argument("--seconds", type=float, default=600, help="how long to watch; Ctrl-C ends it early (default 600)")
    p.add_argument("--kernel-only", action="store_true", help="leave the camera to another script; only read the kernel log")
    p.add_argument("--port", default=None, help="USB port to watch (default: wherever the camera is now, else config's)")
    p.add_argument("--report", action="store_true", help="print every watch so far and how real losses ended, then stop")
    p.add_argument("--sim", action="store_true", help="no hardware: a simulated camera that drops once")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.report:
        return report()
    port = args.port or wrist_usb.port_of() or config.WRIST_USB_PORT
    if args.sim:
        opener, kernel = SimDrop().open, SimKernelLog(port)
    else:
        from mahjong.camera import WristCamera
        opener, kernel = (None if args.kernel_only else WristCamera), KernelLog()
    run = session.Session("usbwatch", ask_setup=False, label=args.label, port=port, kernel_only=args.kernel_only)

    def log(what, **fields):
        run.event(f"watch_{what}", **fields)
        print(f"{time.strftime('%H:%M:%S')} {what}: {fields}")

    print(f"Watching USB port {port} for {args.seconds:.0f}s as '{args.label}'. Ctrl-C ends it.")
    try:
        totals = watch(kernel, port, args.seconds, log, open_camera=opener, lost_s=0.05 if args.sim else None)
        run.event("watch_end", label=args.label, totals=totals)
        print(summary(args.label, totals))
    finally:
        kernel.close()
        run.close()


class SimDrop:
    """Cameras that give a few frames, one of which then stops; the next open works."""

    def __init__(self):
        self.opened = 0

    def open(self):
        self.opened += 1
        return _SimCamera(frames=3 if self.opened == 1 else 10 ** 9)


class _SimCamera:
    def __init__(self, frames):
        self.frames = frames

    def read(self):
        self.frames -= 1
        return self.frames >= 0, None

    def release(self):
        pass


if __name__ == "__main__":
    main()
