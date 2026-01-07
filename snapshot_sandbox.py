import modal
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sbid", type=str, default="sb-REPLACE_ME")
args = parser.parse_args()

sb = modal.Sandbox.from_id(args.sbid)  # reattach to a running sandbox
img = sb.snapshot_filesystem()         # creates an Image snapshot
# SNAPSHOT_IMAGE_ID = im-olrA61MYErROAnQ1Dl5e5f
print("SNAPSHOT_IMAGE_ID =", img.object_id)
