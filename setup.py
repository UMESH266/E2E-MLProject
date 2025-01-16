from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

# Function to collecet libraries
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        content = file_obj.readlines()
        requirements = [library.replace('\n', '') for library in content]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

# Package metadata / package information
setup(
    name='E2EMLProject',
    version='0.0.1',
    description="End to End Machine Learing project structure",
    author='Umesh',
    author_email='umeshgouda143@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)