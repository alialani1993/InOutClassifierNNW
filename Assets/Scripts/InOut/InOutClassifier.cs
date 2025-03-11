using UnityEngine;

public class InOutClassifier : MonoBehaviour
{
    // Neural Network Parameters
    private float[] weightsInputHidden;
    private float[] weightsHiddenOutput;
    private float biasHidden;
    private float biasOutput;

    // Hidden layer neurons
    private float hiddenLayerOutput;

    // Learning rate and threshold
    private const float learningRate = 0.01f;
    private const float threshold = 0.5f;

    void Start()
    {
        // Initialize the neural network parameters
        weightsInputHidden = new float[2] {Random.Range(-1f, 1f), Random.Range(-1f, 1f) }; // Two weights for the input (x and y)
        weightsHiddenOutput = new float[1] {Random.Range(-1f, 1f) }; // One weight for the hidden to output
        biasHidden = Random.Range(-1f, 1f); // Random bias for hidden layer
        biasOutput = Random.Range(-1f, 1f); // Random bias for output layer

        // Start training the network
        //Train(10000);
    }

    public void Train(int epochs)
    {
        // Training data generation (random points between ranges for both x and y)
        float[][] trainingData = new float[1000][];
        int[] labels = new int[1000];

        for (int i = 0; i < trainingData.Length; i++)
        {
            // Random points between ranges
            float x = Random.Range(-7f,7f);
            float y = Random.Range(-4f,4f);
            trainingData[i] = new float[] { x, y };

            // Label 1 if to the left of the line, else label 0
            labels[i] = x <= 0 ? 1 : 0;
        }

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalError = 0f;

            for (int i = 0; i < trainingData.Length; i++)
            {
                float inputX = trainingData[i][0];
                float inputY = trainingData[i][1];
                int label = labels[i];

                // Forward pass
                float predicted = Forward(inputX, inputY);

                // Compute the error
                float error = label - predicted;

                // Backpropagation
                Backpropagate(inputX, inputY, error);

                totalError += Mathf.Abs(error); // Add absolute error to total
            }

            // Log the total error every 100 epochs
            if (epoch % 100 == 0)
            {
                Debug.Log("Epoch " + epoch + " - Total Error: " + totalError);
            }
        }

        Debug.Log("Training Complete");
    }

    // Forward pass: Calculate output of the neural network
    float Forward(float inputX, float inputY)
    {
        // Hidden layer activation (ReLU function)
        hiddenLayerOutput = Mathf.Max(0, inputX * weightsInputHidden[0] + inputY * weightsInputHidden[1] + biasHidden);

        // Output layer (sigmoid activation)
        float output = 1f / (1f + Mathf.Exp(-(hiddenLayerOutput * weightsHiddenOutput[0] + biasOutput)));

        return output;
    }

    // Backpropagation to adjust weights
    void Backpropagate(float inputX, float inputY, float error)
    {
        // Compute the derivative of the sigmoid function
        float outputSigmoid = 1f / (1f + Mathf.Exp(-(hiddenLayerOutput * weightsHiddenOutput[0] + biasOutput)));
        float outputGradient = error * outputSigmoid * (1 - outputSigmoid);  // Derivative of sigmoid

        // Compute hidden layer gradient
        float hiddenGradient = outputGradient * weightsHiddenOutput[0] * (hiddenLayerOutput > 0 ? 1 : 0); // ReLU derivative

        // Update weights and biases
        weightsHiddenOutput[0] += learningRate * outputGradient * hiddenLayerOutput;
        weightsInputHidden[0] += learningRate * hiddenGradient * inputX;
        weightsInputHidden[1] += learningRate * hiddenGradient * inputY;
        biasHidden += learningRate * hiddenGradient;
        biasOutput += learningRate * outputGradient;
    }

    // Classify a point (x, y) as above or below the line y=x
    public string Classify(float x, float y)
    {
        float result = Forward(x, y);
        return result > threshold ? "Left" : "Right";
    }
}
