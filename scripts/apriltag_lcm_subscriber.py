import lcm
from mbot_lcm_msgs.mbot_apriltag_t import mbot_apriltag_t

"""
This scripts subscribe to the MBOT_APRILTAG
We use this program to check if the apriltag publisher works
"""

def apriltag_callback(channel, data):
    msg = mbot_apriltag_t.decode(data)
    print(f"{msg.tag_id}")

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
subscription = lc.subscribe("MBOT_APRILTAG", apriltag_callback)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass
