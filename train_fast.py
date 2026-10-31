"""
neo AI — Fast training (~5 min).
Small model + sample code = fast.
"""

import os
import random
import time

from model import SimpleLM


def get_training_data():
    """Training data — real Python code patterns."""
    return """def hello():
    print('Hello World!')

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        return None
    return a / b

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def delete(self, data):
        if not self.head:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next

    def search(self, data):
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return ' -> '.join(elements)

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

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

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

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if not node.left:
                node.left = TreeNode(val)
                return
            else:
                queue.append(node.left)
            if not node.right:
                node.right = TreeNode(val)
                return
            else:
                queue.append(node.right)

    def inorder(self, node):
        result = []
        if node:
            result = self.inorder(node.left)
            result.append(node.val)
            result = result + self.inorder(node.right)
        return result

def read_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, data):
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def filter_even(numbers):
    return [x for x in numbers if x % 2 == 0]

def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def count_words(text):
    words = text.split()
    return {word: words.count(word) for word in set(words)}

def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)

async def fetch_data(url):
    import asyncio
    await asyncio.sleep(1)
    return {'status': 'ok'}

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

class Logger:
    def __init__(self, name):
        self.name = name

    def info(self, message):
        print(f"[INFO] {self.name}: {message}")

    def error(self, message):
        print(f"[ERROR] {self.name}: {message}")

    def debug(self, message):
        print(f"[DEBUG] {self.name}: {message}")

try:
    result = risky_operation()
except ValueError as e:
    print(f'Error: {e}')
except TypeError as e:
    print(f'Type error: {e}')
finally:
    cleanup()

lambda x: x * 2
[x for x in range(10) if x % 2 == 0]
{x: x**2 for x in range(5)}
{1, 2, 3, 4, 5}

__author__ = 'neo AI'
__version__ = '1.0.0'

import os
import sys
import json
import math
import random

def main():
    print('neo AI running...')
    return 0

if __name__ == '__main__':
    sys.exit(main())
"""


def main():
    print("=" * 50)
    print("  neo AI — Training")
    print("=" * 50)

    dim = 32
    hidden = 128
    epochs = 20
    lr = 0.002

    text = get_training_data()
    print(f"Data size: {len(text)} chars")

    model = SimpleLM(dim=dim, hidden=hidden)
    model.build_vocab(text)

    for epoch in range(epochs):
        start = time.time()
        loss = model.train_epoch(text, lr=lr)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")

    model.save("model.json")
    print("\nTraining complete!")
    print("Run: python app.py")


if __name__ == "__main__":
    main()
