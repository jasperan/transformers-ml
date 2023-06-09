from transformers import pipeline

depth_estimator = pipeline(task="depth-estimation")
preds = depth_estimator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(preds)