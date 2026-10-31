"""
neo AI — Server.
Self-written character-level language model.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from model import SimpleLM


app = Flask(__name__)

model = None
model_path = os.path.join(os.path.dirname(__file__), "model.json")


def load_model():
    """Load model or create demo."""
    global model
    try:
        if os.path.exists(model_path):
            model = SimpleLM(dim=32, hidden=128)
            model.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
    except Exception as e:
        print(f"Error loading model: {e}")

    print("Creating demo model...")
    model = SimpleLM(dim=32, hidden=128)
    demo = """def hello():
    print('Hello!')

def add(a, b):
    return a + b

class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

import os
import sys

def main():
    print('Running...')
    return 0

if __name__ == '__main__':
    sys.exit(main())

__author__ = 'neo AI'
__version__ = '1.0.0'
"""
    model.build_vocab(demo)
    model.save(model_path)
    print("Demo model created")
    return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate code from prompt."""
    global model
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    max_tokens = data.get("max_tokens", 300)
    temperature = data.get("temperature", 0.7)
    top_k = data.get("top_k", 50)

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        result = model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return jsonify({"code": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """Model info."""
    global model
    if model is None:
        return jsonify({"error": "No model"}), 500
    return jsonify({
        "vocab_size": model.vocab_size,
        "dim": model.dim,
        "hidden": model.hidden,
        "chars": len(model.chars),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  neo AI — Self-Written Model")
    print("=" * 50)
    load_model()
    print(f"\nServer: http://0.0.0.0:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
