import math
import threading
from fastapi.testclient import TestClient
from backend.app import create_app


class NativeStub:
    def __init__(self):
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        q = request["questions"]["decision"]
        kind = q["type"]
        keys = (
            list(q["criteria"])
            if kind == "choice"
            else ["false", "true"]
            if kind == "noul"
            else [str(i) for i in range(len(q["criteria"]))]
        )
        probabilities = {k: 1 / len(keys) for k in keys}
        result = dict(
            kind=kind,
            native_option_ids=keys,
            probabilities=probabilities,
            tokens=37,
            truncated=False,
            seconds=0.02,
        )
        if kind == "noul":
            result["value"] = probabilities["true"]
        if kind == "score":
            result["value"] = sum(float(k) * p for k, p in probabilities.items())
        return result


def payload(kind="choice"):
    return {
        "state": "Play jazz.",
        "questions": {
            "decision": {
                "type": kind,
                "instructions": "Choose intent.",
                "criteria": {"music": "Play music.", "weather": None}
                if kind == "choice"
                else {"false": "No", "true": "Yes"}
                if kind == "noul"
                else ["low", "medium", "high"],
            }
        },
    }


def test_three_native_types_and_json_state():
    native = NativeStub()
    client = TestClient(create_app(lambda: native))
    for kind in ["choice", "noul", "score"]:
        data = payload(kind)
        data["state"] = {"request": "Play jazz.", "language": "en"}
        response = client.post("/api/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert (
            result["kind"] == kind
            and result["revision"] == "17cdf36034677e406033fbf97637b4db1984ca70"
        )
        assert math.isclose(sum(result["probabilities"].values()), 1)
    assert native.requests[0]["questions"]["decision"]["criteria"]["weather"] is None
    assert client.get("/api/health").json()["status"] == "ready"


def test_validation_rejects_bad_requests_before_loading():
    def forbidden():
        raise AssertionError("invalid input must not load model")

    client = TestClient(create_app(forbidden))
    bad = [
        {},
        payload() | {"state": ""},
        payload() | {"extra": True},
        payload() | {"state": "x" * 12001},
        payload() | {"state": None},
    ]
    for data in bad:
        assert client.post("/api/predict", json=data).status_code == 422
    for criteria in [
        {},
        {"one": "only"},
        {str(i): "x" for i in range(33)},
        {"": "empty id", "b": "valid"},
    ]:
        data = payload()
        data["questions"]["decision"]["criteria"] = criteria
        assert client.post("/api/predict", json=data).status_code == 422
    data = payload("noul")
    data["questions"]["decision"]["criteria"] = {"yes": "wrong keys"}
    assert client.post("/api/predict", json=data).status_code == 422
    data = payload("score")
    data["questions"]["decision"]["criteria"] = ["one"]
    assert client.post("/api/predict", json=data).status_code == 422
    data = payload()
    data["questions"]["other"] = data["questions"]["decision"]
    assert client.post("/api/predict", json=data).status_code == 422


def test_bad_json_and_oversized_body():
    client = TestClient(create_app(NativeStub))
    assert (
        client.post(
            "/api/predict",
            content="{oops",
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )
    assert client.post("/api/predict", content="x" * 65537).status_code == 413


def test_loading_failure_is_actionable_and_does_not_leak_paths():
    def fail():
        raise RuntimeError("/private/file secret device problem")

    client = TestClient(create_app(fail))
    response = client.post("/api/predict", json=payload())
    assert response.status_code == 503
    assert "setup" in response.json()["detail"].lower()
    assert "/private" not in response.text
    assert client.get("/api/health").json()["status"] == "error"


def test_model_overflow_is_a_user_error():
    class Overflow(NativeStub):
        def predict(self, request):
            raise ValueError("Packed input exceeds qualified context limit")

    client = TestClient(create_app(Overflow))
    response = client.post("/api/predict", json=payload())
    assert (
        response.status_code == 422 and "shorten" in response.json()["detail"].lower()
    )


def test_invalid_native_distribution_is_not_displayed():
    class Broken(NativeStub):
        def predict(self, request):
            r = super().predict(request)
            r["probabilities"]["music"] = float("nan")
            return r

    assert (
        TestClient(create_app(Broken)).post("/api/predict", json=payload()).status_code
        == 502
    )


def test_single_flight_and_load_once():
    entered = threading.Event()
    release = threading.Event()
    loads = []

    class Slow(NativeStub):
        def predict(self, request):
            entered.set()
            release.wait(3)
            return super().predict(request)

    def load():
        loads.append(True)
        return Slow()

    client = TestClient(create_app(load))
    results = []
    thread = threading.Thread(
        target=lambda: results.append(client.post("/api/predict", json=payload()))
    )
    thread.start()
    assert entered.wait(2)
    assert client.post("/api/predict", json=payload()).status_code == 429
    release.set()
    thread.join()
    assert results[0].status_code == 200
    assert client.post("/api/predict", json=payload()).status_code == 200
    assert len(loads) == 1


def test_lifespan_closes_worker_and_releases_model():
    native = NativeStub()
    api = create_app(lambda: native)
    with TestClient(api) as client:
        assert client.post("/api/predict", json=payload()).status_code == 200
        assert api.state.service.model is native
    assert api.state.service.executor._shutdown
    assert api.state.service.model is None


def test_nonfinite_json_state_is_rejected():
    client = TestClient(create_app(NativeStub))
    assert (
        client.post(
            "/api/predict",
            content='{"state":{"amount":1e400},"questions":{"decision":{"type":"choice","instructions":"Choose.","criteria":{"a":"a","b":"b"}}}}',
            headers={"Content-Type": "application/json"},
        ).status_code
        == 422
    )


def test_invalid_native_timing_is_not_returned():
    class Broken(NativeStub):
        def predict(self, request):
            result = super().predict(request)
            result["seconds"] = float("nan")
            return result

    assert (
        TestClient(create_app(Broken)).post("/api/predict", json=payload()).status_code
        == 502
    )


def test_cpu_reports_selected_device_and_enforces_conservative_option_limit():
    native = NativeStub()
    api = create_app(lambda: native, device="cpu")
    with TestClient(api) as client:
        health = client.get("/api/health").json()
        assert health["device"] == "cpu" and health["profile"] == "cpu_fp32"
        assert health["limits"]["options"] == 16
        result = client.post("/api/predict", json=payload()).json()
        assert result["device"] == "cpu" and result["profile"] == "cpu_fp32"
        too_many = payload()
        too_many["questions"]["decision"]["criteria"] = {str(i): "x" for i in range(17)}
        assert client.post("/api/predict", json=too_many).status_code == 422
        assert len(native.requests) == 1


def test_cuda_reports_actual_precision_profile():
    with TestClient(create_app(NativeStub, device="cuda")) as client:
        health = client.get("/api/health").json()
        assert health["device"] == "cuda" and health["profile"] == "cuda_bf16_fp32"
        assert health["limits"]["options"] == 32
        result = client.post("/api/predict", json=payload()).json()
        assert result["device"] == "cuda" and result["profile"] == "cuda_bf16_fp32"


def test_device_is_bound_at_startup_even_if_environment_changes(monkeypatch):
    from backend import runtime

    native = NativeStub()
    selected = []

    def load(*, device):
        selected.append(device)
        return native

    monkeypatch.setattr(runtime, "load_model", load)
    monkeypatch.setenv("CADENCE_DEVICE", "cpu")
    api = create_app()
    monkeypatch.setenv("CADENCE_DEVICE", "cuda")
    with TestClient(api) as client:
        result = client.post("/api/predict", json=payload()).json()
        assert result["device"] == "cpu"
        assert selected == ["cpu"]
