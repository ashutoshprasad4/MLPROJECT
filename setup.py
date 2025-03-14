from setuptools import setup, find_packages
from typing import List

HYPEN = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements file and returns a list of requirements.
    """
    requirements = []
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Strip newline properly

        if HYPEN in requirements:
            requirements.remove(HYPEN)  # Remove "-e ."
    
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Ashutosh",
    author_email="ashutosh.prasad.min22@itbhu.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
