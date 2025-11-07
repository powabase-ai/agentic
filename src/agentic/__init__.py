from celery import Celery, Task
from celery.result import AsyncResult
from flask import Blueprint, Flask, request

from .tasks import add_together

demo = Blueprint("demo", __name__, url_prefix="/demo")


@demo.route("/")
def demo_hello():
    return "hello from demo endpoint"


# to test: curl -X POST -F "a=2" -F "b=4" http://localhost:5000/demo/add
@demo.route("/add", methods=["POST"])
def start_add() -> dict[str, object]:
    """test celery task"""
    a = request.form.get("a", type=int)
    b = request.form.get("b", type=int)
    result = add_together.delay(a, b)
    return {"result_id": result.id}


# to test: curl http://localhost:5000/demo/result/77b01cd4-e441-4afd-8939-30faf1c19582
@demo.route("/result/<id>")
def task_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else None,
    }


def celery_init_app(app: Flask) -> Celery:
    """allow Celery to access Flask application

    e.g. Flask's db connection
    """

    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app


def create_app():
    app = Flask(__name__)

    app.register_blueprint(demo)

    # celery with application factory pattern
    # See https://flask.palletsprojects.com/en/stable/patterns/celery/#application-factory
    app.config.from_mapping(
        CELERY=dict(
            broker_url="redis://redis",
            result_backend="redis://redis",
        ),
    )
    app.config.from_prefixed_env()
    celery_init_app(app)

    return app
