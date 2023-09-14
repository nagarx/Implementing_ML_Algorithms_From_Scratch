
import numpy as np

### 1. Utility Functions ###

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1.0 - y**2

### 2. LSTM Cell Class ###

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        # Ensure both h_prev and x are column vectors
        h_prev = h_prev.reshape(-1, 1)
        x = x.reshape(-1, 1)

        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))

        # Forget gate
        f = sigmoid(np.dot(self.Wf, concat) + self.bf)

        # Input gate
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilda = tanh(np.dot(self.Wc, concat) + self.bc)

        # New cell state
        c = f * c_prev + i * c_tilda

        # Output gate
        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        h = o * tanh(c)

        print("x shape:", x.shape)
        print("h_prev shape:", h_prev.shape)
        print("concat shape:", concat.shape)

        return h, c, f, i, c_tilda, o, concat

    def backward(self, dh_next, dc_next, c_prev, f, i, c_tilda, c, o, h, concat):
        print("dh_next shape:", dh_next.shape)
        print("dc_next shape:", dc_next.shape)
        print("concat shape:", concat.shape)

        # Gradients w.r.t gate outputs
        do = dh_next * tanh(c)
        dc = (dc_next + dh_next * o * dtanh(tanh(c)))
        dc_tilda = dc * i
        di = dc * c_tilda
        df = dc * c_prev

        # Gradients w.r.t gate pre-activations
        df_star = df * dsigmoid(f)
        di_star = di * dsigmoid(i)
        do_star = do * dsigmoid(o)
        dc_tilda_star = dc_tilda * dtanh(c_tilda)

        dz = np.vstack((df_star, di_star, dc_tilda_star, do_star))

        # Gradients w.r.t weights and biases
        print("dz shape:", dz.shape)
        dW = np.dot(dz, concat.T)
        db = np.sum(dz, axis=1, keepdims=True)

        # Gradients w.r.t inputs
        dconcat_f = np.dot(self.Wf.T, dz[:self.hidden_size, :])
        dconcat_i = np.dot(self.Wi.T, dz[self.hidden_size:2 * self.hidden_size, :])
        dconcat_c = np.dot(self.Wc.T, dz[2 * self.hidden_size:3 * self.hidden_size, :])
        dconcat_o = np.dot(self.Wo.T, dz[3 * self.hidden_size:, :])
        dconcat = dconcat_f + dconcat_i + dconcat_c + dconcat_o

        dh_prev = dconcat[:self.hidden_size, :]
        dx = dconcat[self.hidden_size:, :]

        # Gradient w.r.t. previous cell state
        dc_prev = f * dc

        # Split the combined gradients into individual components
        dWf = dW[:self.hidden_size, :]
        dWi = dW[self.hidden_size:2 * self.hidden_size, :]
        dWc = dW[2 * self.hidden_size:3 * self.hidden_size, :]
        dWo = dW[3 * self.hidden_size:, :]

        dbf = db[:self.hidden_size, :]
        dbi = db[self.hidden_size:2 * self.hidden_size, :]
        dbc = db[2 * self.hidden_size:3 * self.hidden_size, :]
        dbo = db[3 * self.hidden_size:, :]

        return dh_prev, dc_prev, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo

### 3. LSTM Layer Class ###

class LSTMLayer:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, input_seq):
        # Initial states
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        # To store intermediate states for backpropagation
        self.h_values = []
        self.c_values = []
        self.f_values = []
        self.i_values = []
        self.c_tilda_values = []
        self.o_values = []
        self.concat_values = []

        for x in input_seq:
            h, c, f, i, c_tilda, o, concat = self.cell.forward(x, h, c)
            self.h_values.append(h)
            self.c_values.append(c)
            self.f_values.append(f)
            self.i_values.append(i)
            self.c_tilda_values.append(c_tilda)
            self.o_values.append(o)
            self.concat_values.append(concat)

        return h, c

    def backward(self, dy, h_values, c_values):
        dh_next = np.zeros_like(h_values[0])
        dc_next = np.zeros_like(c_values[0])

        # To store gradients for all time steps
        dWfs = []
        dWis = []
        dWcs = []
        dWos = []
        dbfs = []
        dbis = []
        dbcs = []
        dbos = []

        # Iterate backwards over the time steps
        for t in reversed(range(len(h_values))):
            dh_next, dc_next, dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo = self.cell.backward(
                dh_next, dc_next,
                c_values[t - 1] if t - 1 >= 0 else np.zeros_like(c_values[0]),
                self.f_values[t],
                self.i_values[t],
                self.c_tilda_values[t],
                self.c_values[t],
                self.o_values[t],
                self.h_values[t],
                self.concat_values[t]
            )

            dWfs.append(dWf)
            dWis.append(dWi)
            dWcs.append(dWc)
            dWos.append(dWo)
            dbfs.append(dbf)
            dbis.append(dbi)
            dbcs.append(dbc)
            dbos.append(dbo)

        # Average gradients over all time steps
        dWf = np.mean(dWfs, axis=0)
        dWi = np.mean(dWis, axis=0)
        dWc = np.mean(dWcs, axis=0)
        dWo = np.mean(dWos, axis=0)
        dbf = np.mean(dbfs, axis=0)
        dbi = np.mean(dbis, axis=0)
        dbc = np.mean(dbcs, axis=0)
        dbo = np.mean(dbos, axis=0)

        print("dh_next shape:", dh_next.shape)
        print("dc_next shape:", dc_next.shape)
        print("concat_values[t] shape:", self.concat_values[t].shape)

        return dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo

### 4. LSTM Model Class ###

class LSTMModel:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.layer = LSTMLayer(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size)
        self.by = np.zeros((output_size, 1))

    def forward(self, input_seq):
        h, _ = self.layer.forward(input_seq)
        y = np.dot(self.Wy, h) + self.by
        return y

    def compute_loss(self, y_pred, y_true):
        return 0.5 * np.sum((y_pred - y_true) ** 2)

    def backward(self, dy, h_values, c_values):
        dWf, dWi, dWc, dWo, dbf, dbi, dbc, dbo = self.layer.backward(dy, h_values, c_values)

        # Update weights and biases using gradients
        self.layer.cell.Wf -= self.learning_rate * dWf
        self.layer.cell.Wi -= self.learning_rate * dWi
        self.layer.cell.Wc -= self.learning_rate * dWc
        self.layer.cell.Wo -= self.learning_rate * dWo
        self.layer.cell.bf -= self.learning_rate * dbf
        self.layer.cell.bi -= self.learning_rate * dbi
        self.layer.cell.bc -= self.learning_rate * dbc
        self.layer.cell.bo -= self.learning_rate * dbo

