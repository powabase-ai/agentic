"""entrypoint for celery worker

See: https://flask.palletsprojects.com/en/stable/patterns/celery/#application-factory
"""

from . import create_app

flask_app = create_app()
celery_app = flask_app.extensions["celery"]
