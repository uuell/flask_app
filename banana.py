from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="haiA2EDLuImQW8OmsYIp"
)

result = CLIENT.infer(your_image.jpg, model_id="banana-ripeness-classification/5")