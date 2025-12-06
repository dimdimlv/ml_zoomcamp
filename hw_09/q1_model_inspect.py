import onnxruntime as ort

model_path = 'hair_classifier_v1.onnx'

session = ort.InferenceSession(model_path)

print("Input nodes:")
for i in session.get_inputs():
    print(f"Name: {i.name}, Type: {i.type}, Shape: {i.shape}")

print("\nOutput nodes:")
for o in session.get_outputs():
    print(f"Name: {o.name}, Type: {o.type}, Shape: {o.shape}")
