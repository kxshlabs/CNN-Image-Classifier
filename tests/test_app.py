import io
from PIL import Image
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CNN_IMAGE_CLASSIFIER" in response.data

def test_predict_route_no_image(client):
    response = client.post("/predict")
    assert response.status_code == 400
    assert response.json["error"] == "No image uploaded"

def test_predict_route_valid_image(client):
    img = Image.new("RGB", (32, 32), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    data = {
        "image": (img_byte_arr, "test.png")
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.json
    assert "top_prediction" in json_data
    assert "confidence" in json_data
    assert len(json_data["all_predictions"]) == 5
