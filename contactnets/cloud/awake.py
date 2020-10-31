import click

from contactnets.cloud import manager

import re

import time

import pdb

@click.command()
@click.option('--name', default=None, type=str, help='Regex for name')
@click.option('--group', default=None, type=str, help='Regex for group label')
def main(name: str, group: str):
    test_name = name is not None
    test_group = group is not None
    assert test_name or test_group 

    name_pattern = re.compile(name + '\Z') if test_name else None
    group_pattern = re.compile(group + '\Z') if test_group else None
    while True:
        print('Checking awake')
        instances = manager.get_instances() 
        for instance in instances:
            name_match = name_pattern.match(instance['name']) if test_name else False
            group_match = ('labels' in instance and \
                           group_pattern.match(instance['labels']['group'])) \
                           if test_group else False
            is_terminated = instance['status'] == 'TERMINATED'
            if (name_match or group_match) and is_terminated:
                print(f"Waking {instance['name']} in {instance['zone']}")
                operation = manager.start_instance(instance)
                operation.execute()

        time.sleep(120)

if __name__ == "__main__": main()
