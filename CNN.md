# CNN Example:

## Example NN that can determine logic gates output:

### Creating the Dataset:
```js
const logicGateData = {
    AND: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [0], [0], [1]] },
    OR: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [1], [1], [1]] },
    XOR: { inputs: [[0, 0], [0, 1], [1, 0], [1, 1]], outputs: [[0], [1], [1], [0]] },
};

function getDataset(gate) {
    const inputs = logicGateData[gate].inputs;
    const outputs = logicGateData[gate].outputs;
    return { inputs, outputs };
}
```

### Defining and Training the Neural Network Model:
```js
async function trainModel(gate) {
    const { inputs, outputs } = getDataset(gate);

    // Convert data to tensors
    const inputTensor = tf.tensor2d(inputs);
    const outputTensor = tf.tensor2d(outputs);

    // Define the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 4, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    // Compile the model
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['accuracy']
    });

    // Train the model
    await model.fit(inputTensor, outputTensor, {
        epochs: 500,
        shuffle: true,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'loss' })
    });

    return model;
}

async function run(gate) {
    const model = await trainModel(gate);

    // Test the model
    const testInputs = logicGateData[gate].inputs;
    const testTensor = tf.tensor2d(testInputs);
    const predictions = model.predict(testTensor);

    predictions.array().then(array => {
        console.log(`Predictions for ${gate} gate:`);
        console.table(array.map(val => Math.round(val[0])));
    });
}
```