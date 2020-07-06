# কোডিং

এখন আমরা আগের চ্যাপ্টারে ডিজাইন করা ডিপ নিউরাল নেটওয়ার্কের একটা প্রোগ্রাম্যাটিক ভার্সন দেখবোঃ

```python
from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 (4 neurons, each with 3 inputs): ")
        print (self.layer1.synaptic_weights)
        print ("    Layer 2 (1 neuron, with 4 inputs):")
        print (self.layer2.synaptic_weights)

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(1)

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(4, 3)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 4)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print ("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print ("Stage 3) Considering a new situation [1, 1, 0] -&amp;amp;amp;amp;amp;amp;gt; ?: ")
    hidden_state, output = neural_network.think(array([1, 1, 0]))
    print (output)
```

আমাদের সিঙ্গেল নিউরন ওয়ালা নেটওয়ার্কের কোডের সাথে অনেক কিছুই মিল আছে এখানে। কারণ, বেশ কিছু ফাংশনালিটি সব রকম নিউরাল নেটওয়ার্কেই লাগে। উপরের প্রোগ্রামে বাড়তি কিছু কোড আছে যেমন - প্রত্যেকবার একটি করে নতুন লেয়ার নেয়ার জন্য একটি ছোট্ট ক্লাস আছে NeuronLayer নামে \(লাইন 4\)। এই ক্লাসের অবজেক্ট তৈরির সময় "নিউরন সংখ্যা" এবং "প্রত্যেকটা নিউরনে আগত ইনপুটের সংখ্যা" ডিফাইন করে দিলেই ওরকম একটা লেয়ার তৈরি হয়ে যাবে। আমাদের ডায়াগ্রাম অনুযায়ী যেমন Layer 1 তৈরি করা হচ্ছে এভাবে।

এরপর আছে NeuralNetwork ক্লাস \( লাইন ৯ \) যেটা অনেকটাই আগের প্রোগ্রামের মতই। তবে গুরুত্বপূর্ণ কিছু পরিবর্তন আছে এই কোডে। যেমন - এখানে নিউরাল নেটওয়ার্ক যখন Layer 2 -এ এসে এরর হিসাব করে তখন সেটা সে Back Propogate করে একদম শুরুতে না নিয়ে বরং Layer 1 এ নিয়ে যায় এবং ওয়েট অ্যাডজাস্ট করে। Layer 2 এর এরর নির্ভর করে লেয়ার ২ এর আউটপুট এবং আসল ট্রেনিং সেট এর আউটপুটের বিয়োগের ফলের উপর \( আগের মতই \) । সাথে অ্যাডজাস্টমেন্ট নির্ধারণের জন্য Sigmoid Derivative \(লেয়ার ২ আউটপুট এর উপর ভিত্তি করে\) এবং ইনপুট হিসেবে লেয়ার ১ এর আউটপুট তো আছেই \(৩৬ এবং ৪৫ নং লাইন খেয়াল করুন\)।

আর Layer 1 এর এরর কিসের উপর নির্ভর করছে সেটা একটু বুঝে শুনে খেয়াল করা উচিৎ। এখানে পার্থক্যটা \(Error\) এমন না যে আসল আউটপুট এবং এক্সপেক্টেড আউটপুট বিয়োগ করেই এররের ধারনা পাওয়া যাবে কারণ Layer 1 এর তো কোন ব্যবহার উপযোগী আউটপুট নাই। বরং এই লেয়ার পরবর্তী লেয়ারের এররের উপর ভূমিকা রাখে। তাই এই লেয়ারের এরর ফ্যাক্টরটা বস্তুত Layer 2 এর ওয়েট এবং এরর ডেরিভেটিভ এর সমন্বয়ের অবস্থাটা। এরপর এই লেয়ারের অ্যাডজাস্টমেন্ট এর জন্য ইনপুট ফ্যাক্টর হিসেবে লাগছে মুল ইনপুট ভ্যালু গুলো, আর আউটপুট হিসেবে এই লেয়ারের আউটপুটের Sigmoid Derivative \( ৪১ এবং ৪৪ নং লাইন \).

train ফাংশনের শেষে এই দুটো লেয়ারের ওয়েট গুলো অ্যাডজাস্ট করা হয়েছে এবং ট্রেনিং লুপ চালিয়ে যাওয়া হয়েছে। এর নিচে থাকা think ফাংশনের কাজ খুব সহজেই বুঝে যাওয়ার কথা কারণ এটা সেই ব্যাসিক নেটওয়ার্কের মতই \(শুধু দুই ধাপের আউটপুট আলাদা করে চিন্তা করছে\)।

উপরের প্রোগ্রামটি রান করালে নিচের মত আউটপুট আসতে পারেঃ

![Screen Shot 2017-05-19 at 7.41.03 PM](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-19-at-7-41-03-pm.png)

পুরো ঘটনাকে ৩টী স্টেজে ভাগ করে নিলে আমরা দেখতে পাই যে - প্রথম ধাপে শুধুমাত্র দুই লেয়ারের প্রত্যেকটি Edge এর ওয়েটকে র‍্যান্ডোমলি নির্ধারণ করা হচ্ছে। পরবর্তী ধাপে পুরো ট্রেনিং প্রসেস শেষে দুটো লেয়ারের প্রত্যেকটি এইজের আপডেটেড এবং অপ্টিমাইজড ওয়েট গুলো দেখতে পাচ্ছি। এবং তৃতীয় ধাপে নিউরাল নেটওয়ার্কে নতুন একটি অচেনা ইনপুট কম্বিনেশন দিয়ে আমরা অউটপুট পাচ্ছি 0.0078 অর্থাৎ সফলভাবে 0 প্রেডিক্ট করতে পারছে আমাদের ডিপ নিউরাল নেটওয়ার্ক :\) :D

