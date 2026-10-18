from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from transitlab.api.app import create_app, get_engine


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    """An app over a freshly written sample feed, loaded once."""
    from transitlab.gtfs.synthetic import sample_feed

    path = tmp_path_factory.mktemp("api") / "feed.zip"
    sample_feed().write(path)

    import os

    os.environ["TRANSITLAB_FEED"] = str(path)
    get_engine.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_engine.cache_clear()
    os.environ.pop("TRANSITLAB_FEED", None)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_feed_summary(client: TestClient) -> None:
    body = client.get("/feed").json()
    assert body["stops"] == 7
    assert body["patterns"] == 5
    assert body["trips"] == 56


def test_stops_and_filter(client: TestClient) -> None:
    body = client.get("/stops").json()
    assert len(body) == 7
    assert {s["stop_id"] for s in body} >= {"S0", "S5"}

    filtered = client.get("/stops", params={"q": "airport"}).json()
    assert [s["stop_id"] for s in filtered] == ["S5"]


def test_journey_with_a_walking_transfer(client: TestClient) -> None:
    body = client.get(
        "/journey", params={"from": "S0", "to": "S5", "time": "06:10", "change_time": 0}
    ).json()
    fastest = body["fastest"]
    assert fastest["arrival"] == "06:40:30"
    assert fastest["transfers"] == 1
    assert [leg["kind"] for leg in fastest["legs"]] == ["ride", "walk", "ride"]
    assert fastest["legs"][0]["route_id"] == "BLUE"
    assert fastest["legs"][1]["seconds"] == 300
    assert fastest["legs"][2]["to_name"] == "Airport"


def test_journey_accepts_hh_mm_and_rejects_nonsense(client: TestClient) -> None:
    assert client.get(
        "/journey", params={"from": "S0", "to": "S1", "time": "06:10"}
    ).status_code == 200
    bad = client.get("/journey", params={"from": "S0", "to": "S1", "time": "tea time"})
    assert bad.status_code == 422


def test_journey_past_midnight(client: TestClient) -> None:
    body = client.get(
        "/journey", params={"from": "S0", "to": "S2", "time": "23:50"}
    ).json()
    assert body["fastest"]["arrival"] == "24:01:30"


def test_journey_when_service_is_over(client: TestClient) -> None:
    body = client.get(
        "/journey", params={"from": "S0", "to": "S5", "time": "23:00"}
    ).json()
    assert body["fastest"] is None
    assert body["alternatives"] == []


def test_unknown_stop_is_404(client: TestClient) -> None:
    response = client.get("/journey", params={"from": "NOWHERE", "to": "S1"})
    assert response.status_code == 404
    assert "NOWHERE" in response.json()["detail"]


def test_isochrone(client: TestClient) -> None:
    body = client.get(
        "/isochrone",
        params={"from": "S0", "time": "06:10", "minutes": 20, "change_time": 0},
    ).json()
    reached = {s["stop_id"]: s["minutes"] for s in body["reached"]}
    assert reached["S0"] == 0
    assert "S5" not in reached  # 30 minutes away, outside a 20-minute budget
    assert body["budget_minutes"] == 20

    wider = client.get(
        "/isochrone",
        params={"from": "S0", "time": "06:10", "minutes": 45, "change_time": 0},
    ).json()
    assert {s["stop_id"] for s in wider["reached"]} >= set(reached)
    assert "S5" in {s["stop_id"] for s in wider["reached"]}


def test_isochrone_minutes_are_validated(client: TestClient) -> None:
    assert client.get("/isochrone", params={"from": "S0", "minutes": 0}).status_code == 422


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/journey" in schema["paths"]
    assert "/isochrone" in schema["paths"]


def test_stops_limit_clears_a_whole_city(client) -> None:
    """A cap below the feed's stop count draws a partial network, silently.

    The UI plots every stop at once, so this is the regression test for the
    5,000 ceiling that was quietly truncating a 9,948-stop feed.
    """
    response = client.get("/stops", params={"limit": 12_000})
    assert response.status_code == 200

    feed = client.get("/feed").json()
    assert len(response.json()) == feed["stops"]
