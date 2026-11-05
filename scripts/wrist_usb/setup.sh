#!/bin/sh
# Once per machine, by a person: sudo scripts/wrist_usb/setup.sh    (--remove takes it all out again)
#  - turns USB autosuspend off for the wrist camera, as 99-realsense-no-autosuspend.rules does for the RealSense
#  - installs the port power-cycle helper, runnable without a password by the user who ran this, and only that helper
set -eu
RULE=/etc/udev/rules.d/99-wristcam-no-autosuspend.rules
HELPER=/usr/local/sbin/wristcam-port-cycle
SUDOERS=/etc/sudoers.d/wristcam-port-cycle

if [ "$(id -u)" != 0 ] || [ -z "${SUDO_USER:-}" ]; then
    echo "Run it with sudo: sudo $0" >&2
    exit 1
fi
if [ "${1:-}" = "--remove" ]; then
    rm -f "$RULE" "$HELPER" "$SUDOERS"
    udevadm control --reload-rules
    echo "Removed. Autosuspend comes back at the next replug or reboot."
    exit 0
fi
here=$(dirname "$(readlink -f "$0")")
install -o root -g root -m 755 "$here/port-cycle.sh" "$HELPER"
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="05a3", ATTR{idProduct}=="9230", TEST=="power/control", ATTR{power/control}="on"' > "$RULE"
tmp=$(mktemp)
printf '%s ALL=(root) NOPASSWD: %s\n' "$SUDO_USER" "$HELPER" > "$tmp"
visudo -cf "$tmp" >/dev/null
install -o root -g root -m 440 "$tmp" "$SUDOERS"
rm -f "$tmp"
udevadm control --reload-rules
udevadm trigger --subsystem-match=usb --attr-match=idVendor=05a3 --attr-match=idProduct=9230
echo "Installed: $RULE, $HELPER, $SUDOERS"
for device in /sys/bus/usb/devices/*; do
    if [ "$(cat "$device/idVendor" 2>/dev/null)" = 05a3 ]; then
        echo "Autosuspend on the camera: $(cat "$device/power/control") (should be: on)"
        echo "What the camera's root hub says about switching power (ganged or per-port can cut it; 'no power switching' can't):"
        lsusb -v -s "$(cat "$device/busnum"):1" 2>/dev/null | grep -iE "power switching" || echo "  (not reported)"
    fi
done
