import subprocess

scene = "small_city"
extract = False
runs = 3

args_slam_srgb = f"configs/rawslam/{scene}/{scene}.yaml"
args_slam_raw_l1_mlp = f"configs/rawslam/{scene}/{scene}_raw_l1_mlp.yaml"
args_slam_raw_l1 = f"configs/rawslam/{scene}/{scene}_raw_l1.yaml"

if extract:
	args = ["--scene", f"{scene}"]
	script = "extract.py"
	print(f"Running {script}...")
	subprocess.run(["python", script] + args, check=True)
	print(f"Finished {script}.")
	print("")

args = [args_slam_srgb, args_slam_raw_l1, args_slam_raw_l1_mlp]

script = "run.py"
for run in range(runs+1):
	for arg in args:
		arg = ["--config", arg]
		print(f"🚀 Running {script} in run num {run}...")
		subprocess.run(["python", script] + arg, check=True)
		print(f"Finished {script}.")
		print("")
