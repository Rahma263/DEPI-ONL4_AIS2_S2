import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x1, x2 = 0.05, 0.10
target_o1 = 0.01
target_o2 = 0.99

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
lr = 0.5

print("\n" + "=" * 65)
print("          ITERATION 1  —  FORWARD PASS")
print("=" * 65)

net_h1 = w1 * x1 + w3 * x2 + b1 * 1
out_h1 = sigmoid(net_h1)

print(f"\n[Step 1]")
print(f"net_h1 = ({w1} * {x1}) + ({w3} * {x2}) + ({b1} * 1)")
print(f"= {net_h1}")

print(f"\n[Step 2]")
print(f"out_h1 = {out_h1}")

net_h2 = w2 * x1 + w4 * x2 + b1 * 1
out_h2 = sigmoid(net_h2)

print(f"\n[Step 3]")
print(f"net_h2 = {net_h2}")
print(f"out_h2 = {out_h2}")

net_o1 = w5 * out_h1 + w7 * out_h2 + b2 * 1
net_o2 = w6 * out_h1 + w8 * out_h2 + b2 * 1
out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

print(f"\n[Step 4]")
print(f"net_o1 = {net_o1}")
print(f"out_o1 = {out_o1}")

print(f"\nnet_o2 = {net_o2}")
print(f"out_o2 = {out_o2}")

E_o1 = 0.5 * (target_o1 - out_o1) ** 2
E_o2 = 0.5 * (target_o2 - out_o2) ** 2
E_total = E_o1 + E_o2

print(f"\n[Step 5]")
print(f"E_o1 = {E_o1}")
print(f"E_o2 = {E_o2}")
print(f"E_total = {E_total}")

print("\n" + "=" * 65)
print("    BACKWARD PASS  (Output Layer)")
print("=" * 65)

dE_dout_o1 = -(target_o1 - out_o1)
dout_o1_dnet = out_o1 * (1 - out_o1)
delta_o1 = dE_dout_o1 * dout_o1_dnet

dE_dout_o2 = -(target_o2 - out_o2)
dout_o2_dnet = out_o2 * (1 - out_o2)
delta_o2 = dE_dout_o2 * dout_o2_dnet

dE_dw5 = delta_o1 * out_h1
w5_new = w5 - lr * dE_dw5

dE_dw6 = delta_o1 * out_h2
w6_new = w6 - lr * dE_dw6

dE_dw7 = delta_o2 * out_h1
w7_new = w7 - lr * dE_dw7

dE_dw8 = delta_o2 * out_h2
w8_new = w8 - lr * dE_dw8

print("\n" + "=" * 65)
print("    BACKWARD PASS  (Hidden Layer)")
print("=" * 65)

dE_dout_h1 = delta_o1 * w5 + delta_o2 * w7
dout_h1_dnet = out_h1 * (1 - out_h1)

dE_dout_h2 = delta_o1 * w6 + delta_o2 * w8
dout_h2_dnet = out_h2 * (1 - out_h2)

dE_dw1 = dE_dout_h1 * dout_h1_dnet * x1
w1_new = w1 - lr * dE_dw1

dE_dw2 = dE_dout_h2 * dout_h2_dnet * x1
w2_new = w2 - lr * dE_dw2

dE_dw3 = dE_dout_h1 * dout_h1_dnet * x2
w3_new = w3 - lr * dE_dw3

dE_dw4 = dE_dout_h2 * dout_h2_dnet * x2
w4_new = w4 - lr * dE_dw4

print("\n" + "=" * 65)
print("    UPDATED WEIGHTS")
print("=" * 65)

for name, old, new in [
    ('W1', w1, w1_new), ('W2', w2, w2_new),
    ('W3', w3, w3_new), ('W4', w4, w4_new),
    ('W5', w5, w5_new), ('W6', w6, w6_new),
    ('W7', w7, w7_new), ('W8', w8, w8_new),
]:
    print(name, old, new)

_h1 = sigmoid(w1_new*x1 + w3_new*x2 + b1)
_h2 = sigmoid(w2_new*x1 + w4_new*x2 + b1)
_o1 = sigmoid(w5_new*_h1 + w7_new*_h2 + b2)
_o2 = sigmoid(w6_new*_h1 + w8_new*_h2 + b2)
E_after_1 = 0.5*(target_o1-_o1)**2 + 0.5*(target_o2-_o2)**2

print("\nError before:", E_total)
print("Error after:", E_after_1)

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60

for i in range(10000):
    net_h1 = w1*x1 + w3*x2 + b1
    net_h2 = w2*x1 + w4*x2 + b1
    out_h1 = sigmoid(net_h1)
    out_h2 = sigmoid(net_h2)

    net_o1 = w5*out_h1 + w7*out_h2 + b2
    net_o2 = w6*out_h1 + w8*out_h2 + b2
    out_o1 = sigmoid(net_o1)
    out_o2 = sigmoid(net_o2)

    delta_o1 = -(target_o1 - out_o1) * out_o1 * (1 - out_o1)
    delta_o2 = -(target_o2 - out_o2) * out_o2 * (1 - out_o2)

    w5 -= lr * delta_o1 * out_h1
    w6 -= lr * delta_o1 * out_h2
    w7 -= lr * delta_o2 * out_h1
    w8 -= lr * delta_o2 * out_h2
    b2 -= lr * (delta_o1 + delta_o2)

    dE_h1 = (delta_o1 * w5 + delta_o2 * w7) * out_h1 * (1 - out_h1)
    dE_h2 = (delta_o1 * w6 + delta_o2 * w8) * out_h2 * (1 - out_h2)

    w1 -= lr * dE_h1 * x1
    w2 -= lr * dE_h2 * x1
    w3 -= lr * dE_h1 * x2
    w4 -= lr * dE_h2 * x2
    b1 -= lr * (dE_h1 + dE_h2)

net_h1 = w1*x1 + w3*x2 + b1
net_h2 = w2*x1 + w4*x2 + b1
out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)
net_o1 = w5*out_h1 + w7*out_h2 + b2
net_o2 = w6*out_h1 + w8*out_h2 + b2
pred_o1 = sigmoid(net_o1)
pred_o2 = sigmoid(net_o2)
E_final = 0.5*(target_o1-pred_o1)**2 + 0.5*(target_o2-pred_o2)**2

print("\nPrediction o1 =", pred_o1)
print("Prediction o2 =", pred_o2)
print("Final Error =", E_final)