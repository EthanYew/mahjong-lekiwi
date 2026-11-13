#!/bin/sh
# Cut the power of the wrist camera's USB port and restore it. Installed root-owned as
# /usr/local/sbin/wristcam-port-cycle by setup.sh, and sudo lets one user run it without a password, so it trusts nothing:
# it takes one port name, and refuses a port with anything on it but the wrist camera (the arm and the RealSense are on
# the same controller), or whose USB3 twin has a device on it. Usage: wristcam-port-cycle 3-8
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
CAMERA=05a3:9230
DEVICES=/sys/bus/usb/devices
OFF_S=3

port=${1:-}
if ! printf '%s' "$port" | grep -Eq '^[0-9]+-[0-9]+(\.[0-9]+)*$'; then
    echo "not a USB port name: '$port'" >&2
    exit 2
fi
hub=${port%.*}
if [ "$hub" = "$port" ]; then    # on the root hub: 3-8 is 3-0:1.0/usb3-port8
    bus=${port%%-*}
    dir=$DEVICES/$bus-0:1.0/usb$bus-port${port#*-}
else                             # on a hub: 3-2.4 is 3-2:1.0/3-2-port4
    dir=$DEVICES/$hub:1.0/$hub-port${port##*.}
fi
if [ ! -f "$dir/disable" ]; then
    echo "no switchable port at $dir" >&2
    exit 3
fi
if [ -e "$dir/device" ]; then
    found=$(cat "$dir/device/idVendor"):$(cat "$dir/device/idProduct")
    if [ "$found" != "$CAMERA" ]; then
        echo "port $port has $found on it, not the wrist camera: leaving it alone" >&2
        exit 4
    fi
fi
if [ -e "$dir/peer/device" ]; then
    echo "port $port's USB3 twin has a device on it: leaving it alone" >&2
    exit 4
fi
trap 'echo 0 > "$dir/disable"' EXIT    # whatever happens, the port ends up on
echo 1 > "$dir/disable"
sleep "$OFF_S"
