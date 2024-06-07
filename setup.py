from setuptools import setup


def read_requirements(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    _requires = []
    _links = []
    for line in lines:
        if line.startswith("git+"):
            _links.append(line)
        else:
            _requires.append(line)
    return _requires, _links


install_requires, dependency_links = read_requirements('requirements.txt')

setup(install_requires=install_requires)
