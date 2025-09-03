import numpy as np
import requests
import torch
from PIL import Image
from torchvision.models import mobilenet_v2
import qai_hub as hub

def compile_and_infer_on_device(api_token, device_name="Samsung Galaxy S24 (Family)", image_url=None):
    # Configure qai-hub
    hub.configure(api_token=api_token)

    # Load pretrained MobileNetV2
    torch_model = mobilenet_v2(pretrained=True)
    torch_model.eval()

    # Trace model for TorchScript
    input_shape = (1, 3, 224, 224)
    example_input = torch.rand(input_shape)
    traced_torch_model = torch.jit.trace(torch_model, example_input)

    # Compile model for on-device runtime
    compile_job = hub.submit_compile_job(
        model=traced_torch_model,
        device=hub.Device(device_name),
        input_specs=dict(image=input_shape),
        options="--target_runtime tflite",
    )
    target_model = compile_job.get_target_model()

    # Profile model on device
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=hub.Device(device_name),
    )

    # Prepare image input
    if image_url is None:
        image_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
    response = requests.get(image_url, stream=True)
    response.raw.decode_content = True
    image = Image.open(response.raw).resize((224, 224))
    input_array = np.expand_dims(
        np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
    )

    # Run inference
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=hub.Device(device_name),
        inputs=dict(image=[input_array]),
    )
    on_device_output = inference_job.download_output_data()
    output_name = list(on_device_output.keys())[0]
    out = on_device_output[output_name][0]
    on_device_probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

    # Download ImageNet class labels
    classes_url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(classes_url, stream=True)
    response.raw.decode_content = True
    categories = [str(s.strip()) for s in response.raw]

    # Get top-5 predictions
    top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
    predictions = [
        {
            "class": int(c),
            "label": categories[c],
            "probability": float(on_device_probabilities[0][c])
        }
        for c in reversed(top5_classes)
    ]

    # Download compiled model
    target_model.download("mobilenet_v2.tflite")

    return {
        "profile_job": profile_job,
        "inference_job": inference_job,
        "predictions": predictions,
        "model_path": "mobilenet_v2.tflite"
    }

# Example usage:
# results = compile_and_infer_on_device(api_token="YOUR_API_TOKEN")
# print(results["predictions"])