from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
from RRTstar_3D_without_orientation import run_experiment

if __name__ == "__main__":
    num_workers = 10  # Number of workers (experiments) to run in parallel
    num_trials_per_worker = 10
    robot_type = "UR5"
    rm_gain = 1e-1
    verbose = False
    dictionary_l = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_experiment, exp_id, num_trials_per_worker, robot_type, rm_gain, verbose)
            for exp_id in range(num_workers)
        ]
        for future in futures:
            res = future.result()
            dictionary_l.append(res)
            exp_id = res["name"]
            computation_time = res["time"]
            end_time = time.time()
            print("time required: ", end_time - start_time)
            np.savez(
                f"experiment_{exp_id}",
                q=res["q"],
                traj_points=res["traj_points"],
                rm_masses=res["rm_masses"],
                time=computation_time,
            )

    import ipdb

    ipdb.set_trace()
