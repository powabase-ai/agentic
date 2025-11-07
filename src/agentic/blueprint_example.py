from flask import Blueprint, request
from celery.result import AsyncResult

from .tasks import add_together

examples = Blueprint("examples", __name__, url_prefix="/examples")


@examples.route("/")
def examples_hello():
    return "hello from examples blueprint"


# to test: curl -X POST -F "a=2" -F "b=4" http://localhost:5000/examples/add
@examples.route("/add", methods=["POST"])
def start_add() -> dict[str, object]:
    """test celery task"""
    a = request.form.get("a", type=int)
    b = request.form.get("b", type=int)
    result = add_together.delay(a, b)
    return {"result_id": result.id}


# to test: curl http://localhost:5000/examples/result/77b01cd4-e441-4afd-8939-30faf1c19582
@examples.route("/result/<id>")
def task_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else None,
    }
