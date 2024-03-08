from invoke import task

import pytest


@task
def flake8(c):
    """Run Flake8 code style checking"""
    c.run('flake8 MultiLabelSegmentation tests setup.py tasks.py', echo=True)


@task(flake8, pytest)
def build(c):
    """Run tests and code analysis"""
    pass
