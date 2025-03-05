def exp_approx(x, terms=10):
    result = 1
    factorial = 1
    power = 1
    for n in range(1, terms):
        factorial *= n
        power *= x
        result += power / factorial
    return result

def tanh(x):
    exp_pos = exp_approx(x)
    exp_neg = exp_approx(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg) if exp_pos + exp_neg != 0 else 0

def tanh_derivative(x):
    return 1 - (tanh(x) ** 2)
def compute_total_error(targets, outputs):
    return 0.5 * sum((t - o) ** 2 for t, o in zip(targets, outputs))

def feedforward(inputs, weights, biases):
    h1_input = (inputs[0] * weights["w1"]) + (inputs[1] * weights["w3"]) + biases["b1"]
    h2_input = (inputs[0] * weights["w2"]) + (inputs[1] * weights["w4"]) + biases["b1"]

    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)

    o1_input = (h1_output * weights["w5"]) + (h2_output * weights["w7"]) + biases["b2"]
    o2_input = (h1_output * weights["w6"]) + (h2_output * weights["w8"]) + biases["b2"]

    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)

    return (h1_output, h2_output), (o1_output, o2_output)
def backpropagation(inputs, hidden_outputs, outputs, targets, weights):
    delta_o1 = (outputs[0] - targets[0]) * tanh_derivative(outputs[0])
    delta_o2 = (outputs[1] - targets[1]) * tanh_derivative(outputs[1])

    delta_h1 = (delta_o1 * weights["w5"] + delta_o2 * weights["w6"]) * tanh_derivative(hidden_outputs[0])
    delta_h2 = (delta_o1 * weights["w7"] + delta_o2 * weights["w8"]) * tanh_derivative(hidden_outputs[1])

    return delta_o1, delta_o2, delta_h1, delta_h2

def update_weights(weights, biases, inputs, hidden_outputs, deltas, learning_rate):
    delta_o1, delta_o2, delta_h1, delta_h2 = deltas
    weights["w5"] -= learning_rate * delta_o1 * hidden_outputs[0]
    weights["w6"] -= learning_rate * delta_o2 * hidden_outputs[0]
    weights["w7"] -= learning_rate * delta_o1 * hidden_outputs[1]
    weights["w8"] -= learning_rate * delta_o2 * hidden_outputs[1]
    weights["w1"] -= learning_rate * delta_h1 * inputs[0]
    weights["w2"] -= learning_rate * delta_h2 * inputs[0]
    weights["w3"] -= learning_rate * delta_h1 * inputs[1]
    weights["w4"] -= learning_rate * delta_h2 * inputs[1]
    biases["b1"] -= learning_rate * delta_h1
    biases["b2"] -= learning_rate * delta_o1

# **إعداد البيانات**
inputs = [0.05, 0.10]  
targets = [0.01, 0.99]  
learning_rate = 0.5
epochs = 10
weights = {
    "w1": 0.15, "w2": 0.20, "w3": 0.25, "w4": 0.30,
    "w5": 0.40, "w6": 0.45, "w7": 0.50, "w8": 0.55
}

biases = {"b1": 0.35, "b2": 0.60}


print("\n🔹 Initial Weights and Biases:")
for key, value in weights.items():
    print(f"{key}: {value:.6f}")
for key, value in biases.items():
    print(f"{key}: {value:.6f}")
print("\n")

total_error_previous = None
for epoch in range(epochs):
    hidden_outputs, outputs = feedforward(inputs, weights, biases)
    total_error = compute_total_error(targets, outputs)
    
    deltas = backpropagation(inputs, hidden_outputs, outputs, targets, weights)
    update_weights(weights, biases, inputs, hidden_outputs, deltas, learning_rate)
    if total_error_previous is not None:
        change = total_error_previous - total_error
        print(f"Epoch {epoch+1}/{epochs}: Total Error = {total_error:.6f} (ΔError = {change:.6f})")
    else:
        print(f"Epoch {epoch+1}/{epochs}: Total Error = {total_error:.6f}")

    total_error_previous = total_error  
print("\n🔹 Updated Weights and Biases After Training:")
for key, value in weights.items():
    print(f"{key}: {value:.6f}")
for key, value in biases.items():
    print(f"{key}: {value:.6f}")
# alaa
asdfg
