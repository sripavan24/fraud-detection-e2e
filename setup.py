from setuptools import find_packages,setup
from typing import List

e_val="-e ."
def get_requirement(file_path:str)->list[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[reg.replace("\n","") for reg in requirements]
        
        if e_val in requirements:
            requirements.remove(e_val)
    return requirements
        
        
setup(
name="MLproject2",
    version="0.0.1",
    author="SriPavan",
    author_email="sripavan@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement("requirement.txt")
)