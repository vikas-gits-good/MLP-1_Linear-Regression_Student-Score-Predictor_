from setuptools import find_packages, setup
from typing import List

Run_text = "-e ."


def get_requirements(File_path: str) -> List[str]:
    """This function returns the list of requirements

    Args:
        File_path (str): File path to 'requirements.txt'

    Returns:
        List[str]: list of libraries and their version number if specified
    """
    requirements = []
    with open(File_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if Run_text in requirements:
            requirements.remove(Run_text)

    return requirements


setup(
    name="ml-project-1",
    version="0.0.1",
    author="Vikas C C",
    author_email="vikas.c.conappa@protonmail.com",
    packages=find_packages(),
    requires=get_requirements(File_path="requirements.txt"),
)
