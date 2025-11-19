"""task definitions for celery

See: https://flask.palletsprojects.com/en/stable/patterns/celery/#application-factory

NOTE: ETL tasks should be defined in client projects, not here.
This is because tasks need access to client-specific models and database connections.
See insurance-demo for an example implementation.
"""

from celery import shared_task


@shared_task
def add_together(a: int, b: int) -> int:
    """Example task"""
    return a + b
