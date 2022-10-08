import functools
import os
import subprocess
import time

import glob
import requests

from fabric import Connection

from inspect import signature


@functools.lru_cache()
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()


def create_tpu(
        name,
        zone,
        type,
        preemptible,
):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
        'Content-Type': 'application/json',
    }

    try:
        status = check_tpu(name, zone)

        if status["state"] not in ["CREATING", "READY"]:
            print("deleting TPU")
            delete_tpu(name, zone)

            while True:
                try:
                    print("deleting check")
                    print(check_tpu(name, zone)["state"])

                    time.sleep(1)
                except:
                    break
    except:
        pass

    params = (
        ('node_id', name),
    )

    data = {"accelerator_type":
                type,
            "runtime_version":
                'v2-alpha',
            "network_config":
                {"enable_external_ips": True},
            }

    if preemptible:
        data["schedulingConfig"] = {"preemptible": True}

    response = requests.post(f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes',
                             headers=headers, params=params, json=data)

    print(response.json())

    return response.status_code == 200


def check_tpu(name, zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.get(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()


def delete_tpu(name, zone):
    headers = {
        'Authorization': f'Bearer {get_bearer()}',
    }

    response = requests.delete(
        f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{zone}/nodes/{name}',
        headers=headers)

    return response.json()


def wait_til(name, zone, state):
    while True:
        ret = check_tpu(name, zone)

        # print("wait_til check")
        # print(ret)

        matches = True
        for k, expected_v in state.items():
            if k not in ret:
                matches = False
                continue
            if ret[k] != expected_v:
                matches = False

        if "error" in ret:
            return False

        if ret["state"] == "TERMINATED":
            return False

        if matches:
            return True

        time.sleep(1)


def get_connection(
    name,
    zone,
):
    info = check_tpu(name, zone)
    outputs = []
    key_filename = os.path.expanduser('~/.ssh/google_compute_engine')

    print('—— get_connection', name, zone, key_filename)
    for i in info["networkEndpoints"]:
        print('networkEndpoint', name, i, i["ipAddress"])
        outputs.append(Connection(
            i["ipAddress"],
            connect_kwargs={ "key_filename": key_filename }
        ))

    return outputs

def start_ray(conn, address, version=1):
    print('— IGNORE START_RAY')

    return

    print('— STOP OLD RAY')
    try:
        conn.run('ray stop -f', hide=False)
    except:
        pass

    time.sleep(3)
    print('—— start_ray version', version, address)

    conn.put("scripts/no_init.sh", "/tmp/ray-tpu.sh")
    conn.sudo('chmod +x /tmp/ray-tpu.sh', hide=False)
    conn.sudo('/tmp/ray-tpu.sh', hide=False)

    print('— copied files')

    time.sleep(3)

    print('——— CONNECTING', address)

    out = conn.run(f"bash /tmp/ray-tpu.sh {address}", hide=False)

    # --include-dashboard False
    # conn.run(f"TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD={32 * 1024**3} ray start --address={address} --resources='" + '{"TPU": 1}\'', hide=False)

    # --load-code-from-local

    # conn.run(f"ray start --address={address} --include-dashboard False --resources='" + '{"TPU": 1}\'')

    # print('— finished start_ray', result)
    # return result


def start_ray_backup(conn, address, version=1):
    conn.sudo('rm -rf *.py')
    conn.sudo('rm -rf mesh_transformer')

    for i in glob.glob("*.py"):
        conn.put(i, "")

    conn.run("mkdir mesh_transformer -p")

    for i in glob.glob("mesh_transformer/*.py"):
        conn.put(i, "mesh_transformer/")

    conn.sudo('python3 setup.py install', hide=False)

    if version == 2:
        conn.put("scripts/init_ray_v2.sh", "/tmp/ray-tpu.sh")
    else:
        conn.put("scripts/init_ray.sh", "/tmp/ray-tpu.sh")

    conn.sudo('chmod +x /tmp/ray-tpu.sh', hide=False)
    conn.sudo('/tmp/ray-tpu.sh', hide=False)

    try:
        conn.run('ray stop -f', hide=False)
    except:
        pass

    time.sleep(1)

    print('——— BACKUP CONECT!!!')

    # --load-code-from-local

    print(
        conn.run(f"ray start --address={address} --include-dashboard=false --resources='" + '{"TPU": 1}\'')
    )
    # conn.run(f"TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD={32 * 1024**3} ray start --address={address} --resources='" + '{"t p u": 1}\' --include-dashboard False', hide=True)
