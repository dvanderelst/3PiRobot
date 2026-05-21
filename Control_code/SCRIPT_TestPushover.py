"""
SCRIPT_TestPushover.py

One-shot smoke test for the Pushover notification path.

Sends a timestamped message via Library.PushOver and prints the HTTP
response (status + body). Use this to check that the phone receives
notifications — if the HTTP response is 200 but nothing arrives on the
phone, the issue is on the device side (silent mode, focus, app
permissions) rather than the code path.
"""
import socket
import time

from Library import PushOver


def main():
    msg = (f"Pushover test {time.strftime('%Y-%m-%d %H:%M:%S')} "
           f"from {socket.gethostname()}")
    print(f"Sending: {msg}")
    response = PushOver.send(msg)
    status = response.status
    body = response.read().decode("utf-8", errors="replace")
    print(f"HTTP {status}")
    print(f"Body : {body}")
    if status == 200:
        print("\nServer accepted the message. If it doesn't arrive on the "
              "phone, check device-side: silent mode, focus, Pushover app "
              "notification permissions, or Quiet Hours in the Pushover app.")
    else:
        print("\nServer rejected the message. Most common cause: a bad "
              "token / user key in Library/PushOver.py.")


if __name__ == "__main__":
    main()
