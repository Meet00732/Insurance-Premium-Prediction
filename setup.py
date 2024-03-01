from setuptools import find_packages, setup
from typing import List

requirement_file = "requirements.txt"
REMOVE_PACKAGE = "-e ."

def get_requirement() -> List[str]:
    with open(requirement_file) as req_file:
        req_list = req_file.readline()
    req_list = [req_name.replace("\n", "") for req_name in req_list]

    if REMOVE_PACKAGE in req_list:
        req_list.remove(REMOVE_PACKAGE)
    return req_list

setup(
    name='Insurance',
    version='0.0.1',
    description='Insurance Premium Prediction project',
    author='Meet Patel',
    author_email='patelmit640@gmail.com',
    packages=find_packages(),
    install_reqires = get_requirement()
)