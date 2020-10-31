from pprint import pprint

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

import os.path as op
import subprocess
import json

from contactnets.utils import file_utils, dirs

from typing import *

import time
import webbrowser
import tabulate
import git

import re

import click
import pprint

import pdb

credentials = GoogleCredentials.get_application_default()
compute = discovery.build('compute', 'v1', credentials=credentials)
project = 'contact-learning'

Instance = Dict[str, Any]

def wait_for_operation(zone, operation):
    if not isinstance(operation, list):
        operation = [operation]

    operations = [op.execute() for op in operation]
    while True:
        for op in operations:
            result = compute.zoneOperations().get(
                project=project,
                zone=zone,
                operation=op['name']).execute()

            if result['status'] == 'DONE':
                if 'error' in result:
                    raise Exception(result['error'])
                return result

        time.sleep(0.05)

def get_instances() -> List[Instance]:
    request = compute.instances().aggregatedList(project=project)

    instances = []
    while request is not None:
        response = request.execute()

        items = response['items']

        for zone in items:
            if 'instances' in items[zone]:
                instances += items[zone]['instances']

        request = compute.instances().list_next(previous_request=request, previous_response=response)

    for instance in instances:
        instance['zone'] = instance['zone'].split('/')[-1]
        if instance['status'] == 'RUNNING':
            instance['ip'] = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']

    return instances

def get_instance(name: str) -> Instance:
    instances = get_instances()
    for instance in instances:
        if instance['name'] == name:
            return instance

def create_instance(name: str, zone: str, group:str=None,
                    preemptible=False, sweep=False, e2e=False, d3=False, socp=False, generate=True):
    image_response = compute.images().get(project=project,image='learning-base-v7').execute()
    source_disk_image = image_response['selfLink']

    machine_type = f'zones/{zone}/machineTypes/e2-highmem-2'
    script = 'startup'
    if sweep: script += '-sweep'
    script += '.sh'

    script = open(op.join(op.dirname(__file__), script), 'r').read()
    cone_options = '--socp ' if socp else ''

    if sweep:
        sweep_options = ''
        if e2e: sweep_options += ' --e2e'
        sweep_options += ' --generate' if generate else ' --data'
        sweep_options = cone_options + sweep_options
        script = script.replace('{sweep_options}', sweep_options)
    else:
        train_options = '' if sweep else '--epochs 1000'
        if e2e: train_options += ' --e2e'
        train_options = cone_options + train_options
        script = script.replace('{train_options}', train_options)

        script = script.replace('{do_gen}', 'true' if generate else 'false')

        script = script.replace('{data_options}', 'multi 200 --center --zrot --pullback')

        gen_options = '--runs 200' if d3 else '--runs 60'
        gen_options = cone_options + gen_options
        script = script.replace('{gen_options}', gen_options)

    experiment = 'block3d' if d3 else 'block2d'
    script = script.replace('{experiment}', experiment)

    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha
    script = script.replace('{hash}', commit_hash)

    config = {
            'name': name,
            'machineType': machine_type,
            'disks': [{
                'boot': True,
                'autoDelete': True,
                'initializeParams': { 'sourceImage': source_disk_image }
            }],
            # Specify a network interface with NAT to access the public internet.
            'networkInterfaces': [{
                'network': 'global/networks/default',
                'accessConfigs': [
                    {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                ]
            }],
            # Allow the instance to access cloud storage and logging.
            'serviceAccounts': [{
                'email': 'default',
                'scopes': [
                    'https://www.googleapis.com/auth/devstorage.read_write',
                    'https://www.googleapis.com/auth/logging.write'
                ]
            }],
            'metadata': {
                'items': [{
                    'key': 'startup-script',
                    'value': script
                }]
            },
            'scheduling': {
                'preemptible': preemptible
            },
            'tags': {
                'items': ['tensorboard-server']
            }}

    if group is not None:
        config['labels'] = [{'group': group}]

    print(script)
    return compute.instances().insert(
            project=project,
            zone=zone,
            body=config)

def delete_instance(instance: Instance):
    operation = compute.instances().delete(project=project,
                                           zone=instance['zone'],
                                           instance=instance['name'])
    return operation

def start_instance(instance: Instance):
    operation = compute.instances().start(project=project,
                                          zone=instance['zone'],
                                          instance=instance['name'])
    return operation

def stop_instance(instance: Instance):
    operation = compute.instances().stop(project=project,
                                         zone=instance['zone'],
                                         instance=instance['name'])
    return operation

def fetch_file_command(instance: Instance, contactnets_path: str,
                        target_path: str, recurse=False) -> List[str]:
    command = ['gcloud', 'compute', 'scp',
            f"{instance['name']}:/home/samuel/SoPhTER/{contactnets_path}",
            target_path, '--zone', instance['zone']]

    if recurse:
        command.insert(3, '--recurse')

    return command

def execute_fetch_commands(commands: List[str]) -> List[int]:
    ops = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) \
           for command in commands]
    exit_codes = [op.wait() for op in ops]

    return exit_codes

def fetch_states() -> List[List[str]]:
    instances = get_instances()
    cmds = []
    for instance in instances:
        instance_dir = dirs.results_path('cloud', instance['name'])
        file_utils.ensure_created_directory(instance_dir)
        file_utils.create_empty_directory(instance_dir + '/run')
        file_utils.write_file(instance_dir + '/instance.dict',
                              pprint.pformat(instance, indent=4))

        if instance['status'] == 'RUNNING':
            cmds.append(fetch_file_command(instance, 'out/variables.json',
                                           instance_dir + '/run/variables.json'))
            cmds.append(fetch_file_command(instance, 'out/params/experiment.json',
                                           instance_dir + '/run/experiment.json'))
            cmds.append(fetch_file_command(instance, 'out/params/training.json',
                                           instance_dir + '/run/training.json'))

    return cmds

def fetch_out(instance: Instance, tensorboard=False) -> List[List[str]]:
    instance_dir = dirs.results_path('cloud', instance['name'])
    file_utils.ensure_created_directory(instance_dir)
    file_utils.create_empty_directory(instance_dir + '/out')
    file_utils.write_file(instance_dir + '/instance.dict',
                          pprint.pformat(instance, indent=4))

    cmds = []
    if instance['status'] == 'RUNNING':
        # Can't just recurse on out because tensorboard directory might be huge
        cmds.append(fetch_file_command(instance, '/out/best',
                                       instance_dir + '/out/best', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/data',
                                       instance_dir + '/out/data', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/params',
                                       instance_dir + '/out/params', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/renders',
                                       instance_dir + '/out/renders', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/trainer.pt',
                                       instance_dir + '/out/trainer.pt', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/variables.json',
                                       instance_dir + '/out/variables.json', recurse=True))
        cmds.append(fetch_file_command(instance, '/out/variables.pickle',
                                       instance_dir + '/out/variables.pickle', recurse=True))
        if tensorboard:
            cmds.append(fetch_file_command(instance, '/out/tensorboard',
                                           instance_dir + '/out/tensorboard', recurse=True))

    return cmds

def fetch_sweep(instance: Instance, tensorboard=False) -> List[List[str]]:
    instance_dir = dirs.results_path('cloud', instance['name'])
    file_utils.ensure_created_directory(instance_dir)
    file_utils.write_file(instance_dir + '/instance.dict',
                          pprint.pformat(instance, indent=4))

    cmds = []
    if instance['status'] == 'RUNNING':
        if tensorboard:
            file_utils.create_empty_directory(instance_dir + '/sweep')
            cmds.append(fetch_file_command(instance, '/results/sweep',
                                           instance_dir + '/sweep', recurse=True))
        else:
            cmds.append(fetch_file_command(instance, '/results/sweep-no-tb',
                                           instance_dir + '/sweep-no-tb', recurse=True))

    return cmds

@click.group()
def cli():
    pass

@cli.command('list')
def list_instances_cmd() -> None:
    instances = get_instances()
    for instance in instances:
        pprint.pprint(instance)

@cli.command('create')
@click.argument('name')
@click.option('--zone', default='us-central1-a')
@click.option('--num', default=1)
@click.option('--group', default=None, type=str)
@click.option('--preemptible/--regular', default=True)
@click.option('--wait/--no_wait', default=True)
@click.option('--sweep/--no_sweep', default=False)
@click.option('--e2e/--structured', default=False)
@click.option('--d3/--d2', default=False)
@click.option('--socp/--cone', default=False)
@click.option('--generate/--data', default=True)
def create_instance_cmd(name: str, zone: str, num: int, group: str,
        preemptible: bool, wait: bool, sweep: bool, e2e: bool, d3: bool, socp: bool, generate: bool) -> None:
    repo = git.Repo(search_parent_directories=True)

    commits_ahead = sum(1 for _ in repo.iter_commits('origin/master..master'))
    if commits_ahead > 0:
        if not click.confirm(f'You are {commits_ahead} commits ahead of master, continue?'):
            raise RuntimeError('Make sure you have pushed commits!')

    changed_files = [item.a_path for item in repo.index.diff(None)]
    if len(changed_files) > 0:
        print('Uncommitted changes to:')
        print(changed_files)
        if not click.confirm(f'Continue?'):
            raise RuntimeError('Make sure you have committed changes!')

    if num == 1:
        operation = create_instance(name, zone, group=group,
                preemptible=preemptible, sweep=sweep, e2e=e2e, d3=d3, socp=socp, generate=generate)
    else:
        operation = [create_instance(name + '-' + str(i), zone, group=group,
            preemptible=preemptible, sweep=sweep, e2e=e2e, d3=d3, socp=socp, generate=generate) for i in range(num)]

    if wait:
        wait_for_operation(zone, operation)
    else:
        operation.execute()

@cli.command('delete')
@click.argument('regex')
def delete_instance_cmd(regex: str) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    for instance in instances:
        if pattern.match(instance['name']):
            print(f"Deleting {instance['name']}")
            delete_instance(instance).execute()

@cli.command('start')
@click.argument('regex')
def start_instance_cmd(regex: str) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    for instance in instances:
        if pattern.match(instance['name']):
            print(f"Starting {instance['name']}")
            start_instance(instance).execute()

@cli.command('stop')
@click.argument('regex')
def stop_instance_cmd(regex: str) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    for instance in instances:
        if pattern.match(instance['name']):
            print(f"Stopping {instance['name']}")
            stop_instance(instance).execute()

@cli.command('states')
def fetch_states_cmd() -> None:
    exit_codes = execute_fetch_commands(fetch_states())
    instances = get_instances()

    data = [['Name', 'Status', 'Epochs', 'Runs']]
    for instance in instances:
        instance_dir = dirs.results_path('cloud', instance['name'])
        run_dir = dirs.results_path('cloud', instance['name'], 'run')
        if op.isdir(run_dir) and not file_utils.num_files(run_dir) == 0:
            with open(run_dir + '/variables.json', 'rb') as file:
                variables = json.load(file)
            with open(run_dir + '/training.json', 'rb') as file:
                training = json.load(file)
            with open(run_dir + '/experiment.json', 'rb') as file:
                experiment = json.load(file)

            data.append([instance['name'], instance['status'], len(variables) - 1, experiment['run_n']])
        else:
            data.append([instance['name'], instance['status'], '-', '-'])

    print(tabulate.tabulate(data))

@cli.command('fetch-out')
@click.argument('regex')
@click.option('--tensorboard/--no-tensorboard', default=False)
def fetch_out_cmd(regex: str, tensorboard: bool) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    cmds = []
    for instance in instances:
        if pattern.match(instance['name']) and instance['status'] == 'RUNNING':
            print(f"Fetching {instance['name']}")
            cmds.extend(fetch_out(instance, tensorboard))
    print(execute_fetch_commands(cmds))

@cli.command('fetch-sweep')
@click.argument('regex')
@click.option('--tensorboard/--no-tensorboard', default=False)
def fetch_sweep_cmd(regex: str, tensorboard: bool) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    cmds = []
    for instance in instances:
        if pattern.match(instance['name']) and instance['status'] == 'RUNNING':
            print(f"Fetching {instance['name']}")
            cmds.extend(fetch_sweep(instance, tensorboard))
    print(execute_fetch_commands(cmds))

@cli.command('view')
@click.argument('regex')
@click.option('--single/--sweep', default=True)
def view_instance_cmd(regex: str, single: bool) -> None:
    instances = get_instances()
    pattern = re.compile(regex + '\Z')
    for instance in instances:
        if pattern.match(instance['name']) and instance['status'] == 'RUNNING':
            print(f"Opening {instance['name']}")
            ip = instance['ip']
            port = 6006 if single else 6007
            webbrowser.open(f'http://{ip}:{port}/#images&_smoothingWeight=0')

if __name__ == '__main__': cli()
