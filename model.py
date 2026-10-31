"""
neo AI — Character-level Language Model.
Pure Python, no ML libraries.
"""

import math
import random

class SimpleLM:
    """Neural network with backprop + Adam (pure Python)."""

    def __init__(self, dim=96, hidden=384):
        self.dim = dim
        self.hidden = hidden
        self.emb = None
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.stoi = None
        self.itos = None
        self.vocab_size = 0
        self.chars = []
        self._adam = {}

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.itos = {i: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.chars = chars
        self._init_weights()
        print(f"Vocab: {self.vocab_size} chars, dim={self.dim}, hidden={self.hidden}")

    def _init_weights(self):
        self.emb = [[random.gauss(0, 0.02) for _ in range(self.dim)] for _ in range(self.vocab_size)]
        s1 = math.sqrt(2.0 / self.dim)
        self.w1 = [[random.gauss(0, s1) for _ in range(self.hidden)] for _ in range(self.dim)]
        self.b1 = [0.0] * self.hidden
        s2 = math.sqrt(2.0 / self.hidden)
        self.w2 = [[random.gauss(0, s2) for _ in range(self.vocab_size)] for _ in range(self.hidden)]
        self.b2 = [0.0] * self.vocab_size

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '?') for t in tokens])

    def forward(self, token):
        x = self.emb[token][:]
        h = [0.0] * self.hidden
        for j in range(self.hidden):
            s = self.b1[j]
            for i in range(self.dim):
                s += x[i] * self.w1[i][j]
            h[j] = max(0.0, s)
        logits = [0.0] * self.vocab_size
        for j in range(self.vocab_size):
            s = self.b2[j]
            for i in range(self.hidden):
                s += h[i] * self.w2[i][j]
            logits[j] = s
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        cache = {
            'token': token,
            'x': x,
            'h': h,
            'logits': logits,
            'probs': probs,
        }
        return probs, cache

    def backward_and_update(self, cache, target_token, lr=0.001, eps=1e-8, beta1=0.9, beta2=0.999):
        probs = cache['probs']
        x = cache['x']
        h = cache['h']
        dlogits = probs[:]
        dlogits[target_token] -= 1.0
        dw2 = [[0.0] * self.vocab_size for _ in range(self.hidden)]
        db2 = dlogits[:]
        for i in range(self.hidden):
            for j in range(self.vocab_size):
                dw2[i][j] = h[i] * dlogits[j]
        dh = [0.0] * self.hidden
        for i in range(self.hidden):
            s = 0.0
            for j in range(self.vocab_size):
                s += self.w2[i][j] * dlogits[j]
            dh[i] = s
        for i in range(self.hidden):
            if h[i] <= 0:
                dh[i] = 0.0
        dw1 = [[0.0] * self.hidden for _ in range(self.dim)]
        db1 = dh[:]
        for i in range(self.dim):
            for j in range(self.hidden):
                dw1[i][j] = x[i] * dh[j]
        dx = [0.0] * self.dim
        for i in range(self.dim):
            s = 0.0
            for j in range(self.hidden):
                s += self.w1[i][j] * dh[j]
            dx[i] = s
        d_emb = [[0.0] * self.dim for _ in range(self.vocab_size)]
        for d in range(self.dim):
            d_emb[cache['token']][d] = dx[d]
        max_grad = 5.0
        for i in range(self.dim):
            for j in range(self.hidden):
                dw1[i][j] = max(-max_grad, min(max_grad, dw1[i][j]))
        for i in range(self.hidden):
            db1[i] = max(-max_grad, min(max_grad, db1[i]))
            for j in range(self.vocab_size):
                dw2[i][j] = max(-max_grad, min(max_grad, dw2[i][j]))
        for j in range(self.vocab_size):
            db2[j] = max(-max_grad, min(max_grad, db2[j]))
        bias_correction1 = 1 - beta1 ** 100
        bias_correction2 = 1 - beta2 ** 100

        def adam_update(param, grad, key):
            if isinstance(param, list) and len(param) > 0 and isinstance(param[0], list):
                rows, cols = len(param), len(param[0])
                is_2d = True
            else:
                rows = len(param)
                cols = None
                is_2d = False
            if key not in self._adam:
                if is_2d:
                    self._adam[key] = [[0.0] * cols for _ in range(rows)]
                    self._adam[key + '_m'] = [[0.0] * cols for _ in range(rows)]
                    self._adam[key + '_v'] = [[0.0] * cols for _ in range(rows)]
                else:
                    self._adam[key] = [0.0] * rows
                    self._adam[key + '_m'] = [0.0] * rows
                    self._adam[key + '_v'] = [0.0] * rows
            m = self._adam[key + '_m']
            v = self._adam[key + '_v']
            if is_2d:
                for i in range(rows):
                    for j in range(cols):
                        g = grad[i][j]
                        m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                        v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                        mb = m[i][j] / bias_correction1
                        vb = v[i][j] / bias_correction2
                        param[i][j] -= lr * mb / (math.sqrt(vb) + eps)
            else:
                for i in range(rows):
                    g = grad[i]
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    mb = m[i] / bias_correction1
                    vb = v[i] / bias_correction2
                    param[i] -= lr * mb / (math.sqrt(vb) + eps)

        adam_update(self.w1, dw1, 'w1')
        adam_update(self.b1, db1, 'b1')
        adam_update(self.w2, dw2, 'w2')
        adam_update(self.b2, db2, 'b2')
        adam_update(self.emb, d_emb, 'emb')

    def generate(self, prompt, max_tokens=100, temperature=0.7, top_k=20):
        tokens = self.encode(prompt[-30:])
        output = list(tokens)
        for _ in range(max_tokens):
            probs, _ = self.forward(output[-1])
            if temperature > 0:
                indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:top_k]
                total = sum(p for _, p in indexed)
                probs = [0.0] * len(probs)
                for idx, p in indexed:
                    probs[idx] = p / total
            r = random.random()
            cum = 0.0
            next_token = len(probs) - 1
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    next_token = i
                    break
            output.append(next_token)
            if next_token == self.stoi.get('\n', 0) and len(output) > 10:
                if output[-3:].count(next_token) > 1:
                    break
        return self.decode(output)

    def train_epoch(self, text, lr=0.002, chunk_size=50):
        tokens = self.encode(text)
        total_loss = 0.0
        count = 0
        for i in range(1, len(tokens)):
            curr = tokens[i - 1]
            target = tokens[i]
            probs, cache = self.forward(curr)
            total_loss -= math.log(max(probs[target], 1e-10))
            count += 1
            self.backward_and_update(cache, target, lr=lr)
            if count % 1000 == 0:
                import gc
                gc.collect()
        return total_loss / max(count, 1)

    def save(self, path):
        import json
        data = {
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()},
            'dim': self.dim,
            'hidden': self.hidden,
            'emb': self.emb,
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Saved to {path}")

    def load(self, path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.stoi = data['stoi']
        self.itos = {int(k): v for k, v in data['itos'].items()}
        self.dim = data['dim']
        self.hidden = data['hidden']
        self.vocab_size = len(self.stoi)
        self.chars = sorted(self.stoi.keys())
        self.emb = data['emb']
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        print(f"Loaded from {path}")
