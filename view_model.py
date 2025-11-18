import mujoco
import mujoco.viewer
import time
import os

# --- Configuration ---
# Replace with the correct path to your XML file
XML_FILENAME = "assets/humanoid_180_75.xml"
# XML_FILENAME = "assets/my_humanoid.xml"
SCRIPT_DIR = os.path.dirname(__file__) # Gets directory where script is run
XML_PATH = os.path.join(SCRIPT_DIR, XML_FILENAME) # Assumes XML is in same dir

if not os.path.exists(XML_PATH):
    print(f"Error: XML file not found at {XML_PATH}")
    exit()
# --------------------


# Load the model
try:
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    print(f"Successfully loaded model: {XML_FILENAME}")
    print(f"  Number of bodies: {m.nbody}")
    print(f"  Number of joints: {m.njnt}")
    print(f"  Number of geoms: {m.ngeom}")
    print(f"  Number of actuators: {m.nu}")
    print(f"  Number of sensors: {m.nsensor}")

except Exception as e:
    print(f"Error loading XML file: {e}")
    exit()

# Launch the viewer
print("\nLaunching viewer...")
print("Viewer Controls (Common):")
print("  Spacebar: Pause/Resume")
print("  Backspace: Reset")
print("  S: Slow Motion")
print("  I: Info Overlay")
print("  F: Contact Forces")
print("  Right-Click+Drag: Rotate Camera")
print("  Ctrl+Left-Click+Drag (on model): Apply Force")
print("  Scroll Wheel: Zoom")
print("  H: Help (often)")

# launch_passive runs the viewer in a separate thread/process
# allowing the main script to potentially continue or just wait.
# It handles the simulation loop internally.
mujoco.viewer.launch(m, d)

# The script will effectively pause here until the viewer window is closed.
print("\nViewer closed.")