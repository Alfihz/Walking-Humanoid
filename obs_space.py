import mujoco


# Temporary load to check sizes - THIS IS INEFFICIENT, DO IT PROPERLY!
# A better way is to define _get_obs first and derive size from its output shape

# Or manually calculate based on XML structure and _get_obs components.
# Example calculation (adjust to your actual _get_obs):

xml_file = "assets/humanoid_180_75.xml"

temp_model = mujoco.MjModel.from_xml_path(xml_file)
qpos_dim = temp_model.nq
qvel_dim = temp_model.nv
# cfrc_ext_dim = 6 * temp_model.nbody # Example, if you include cfrc_ext
sensor_dim = temp_model.nsensordata # If you include sensor data
com_dim = temp_model.data.cvel
comin_dim = temp_model.data.cinert
act_dim = temp_model.nu

# obs_size = (qpos_dim - 2) + qvel_dim + cfrc_ext_dim + sensor_dim # Example based on Humanoid-v4 structure



# Let's guess based on previous discussions, needs VERIFICATION
# qpos (33) - root(xy) (2) = 31
# qvel (32)
# sensor (accel(3) + gyro(3) + touch(2)) = 8
# Total = 31 + 32 + 8 = 71 (This is just a GUESS based on your likely XML)

print(qpos_dim, qvel_dim, sensor_dim, com_dim, comin_dim, act_dim)