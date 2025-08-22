import serial
import serial.tools.list_ports as lp
import time


def force_break(port, baud=115200):
    with serial.Serial(port, baud, timeout=0.3) as s:
        # toggle DTR/RTS like IDEs do
        s.dtr = False;
        s.rts = False
        time.sleep(0.05)
        s.dtr = True;
        s.rts = True
        time.sleep(0.05)
        # send Ctrl-C twice and Ctrl-D
        s.write(b'\x03\x03')  # break
        time.sleep(0.2)
        s.write(b'\x04')  # soft reset
        time.sleep(0.4)


def scan_for_ports():
    ports = []
    for p in lp.comports():
        ports.append({
            "device": p.device,
            "description": p.description or "",
            "manufacturer": getattr(p, "manufacturer", None),
            "product": getattr(p, "product", None),
            "vid": f"{p.vid:04X}" if getattr(p, "vid", None) is not None else None,
            "pid": f"{p.pid:04X}" if getattr(p, "pid", None) is not None else None,
            "serial_number": getattr(p, "serial_number", None),
        })
    # stable order: by device path then description
    ports.sort(key=lambda d: (d["device"] or "", d["description"]))
    return ports


def select_robot_ports(ports):
    robot = "Pololu"
    selected_ports = []
    for p in ports:
        keys = p.keys()
        if 'description' in keys and robot in p["description"]:
            selected_ports.append(p['device'])
    return selected_ports
