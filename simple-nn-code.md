## “Talk is cheap. Show me the code.”  
এই বানীটি কার, তার নাম জানলে নিচে কমেন্ট করতে পারেন। এই যে আমরা নিউরাল নেটওয়ার্ক এর কাহিনীকে ফলো করে সেরকম নীতিতে আমাদের বাস্তবের কিছু সমস্যা সমাধানের জন্য একটা পদ্ধতি নিয়ে চিন্তা করলাম সেটা তো আর খাতা কলমে করে কুলাবে না। এই কাজটা কম্পিউটার দিয়ে করালে খুব দ্রুত আমাদের উদ্দেশ্য পুড়ন হবে। আর সবাই জানে, কম্পিউটারকে দিয়ে ইচ্ছামত কামলা খাটুনি খাটিয়ে নেয়া যায়। শুধুমাত্র তাকে তার ভাষায় আদেশ দিতে হবে। এর নাম নাকি আবার কম্পিউটার প্রোগ্রাম। তো, কি আর করা, লিখে ফেলি; কম্পিউটার বোঝে এবং আমাদের লিখতে সহজ এমন একটা ভাষায় একটা প্রোগ্রাম, যার মাধ্যমে বস্তুতপক্ষে আমরা উপড়ে আলোচ্য কাজ গুলোকেই করবো।

যদি আপনার পাইথন প্রোগ্রামিং ল্যাঙ্গুয়েজে ভালো দখল থাকে তাহলে আপনার জন্য ডাটা সায়েন্স, মেশিন লার্নিং এবং ডিপ লার্নিং নিয়ে কাজ করা সহজ হয়ে যায়। আমরা নিচে একটা পূর্ণ প্রোগ্রাম দেখবো যার মাধ্যমে তিনটি ইনপুট ওয়ালা একটি সিঙ্গেল নিউরন তৈরি করা হয়েছে এবং সেই ইনপুট এইজ গুলোতে প্রথমে কিছু র‍্যান্ডোম ওয়েট সেট করা হয়েছে। এরপর ওই নিউরনে ট্রেনিং ডাটাসেট অর্থাৎ কিছু ইনপুট row এবং row সাপেক্ষে একটি করে আউটপুট দিয়ে দেয়া হয়েছে। Sigmoid Function ব্যবহার করে নিউরনের চিন্তা অনুযায়ী আউটপুট বের করা হয়েছে। সত্যিকারের আউপুট এবং নিউরনের হিসাব করে বের করা আউটপুটের তুলনা করে এরর চেক করা হয়েছে। ১০০০০ বার লুপ চালিয়ে (ট্রেনিং করিয়ে) উপড়ে আলোচ্য ওয়েট অ্যাডজাস্ট করার সূত্র দিয়ে প্রত্যেক লুপের মধ্যে একবার করে ওয়েট অ্যাডজাস্ট করা হয়েছে। সবশেষে একই নিউরনে নতুন একটি ডাটাসেট দিয়ে তার আউপুট জানতে চাওয়া হয়েছে। যদি সে আমাদের ধারনা করা আউপুটকেই আউটপুট হিসেবে দিতে পারে তাহলে বলা যায় যে, এই সিঙ্গেল নিউরন ওয়ালা নেটওয়ার্কটি ৪টি ট্রেনিং ডাটাসেট থেকেই প্যাটার্ন খুঁজে নিতে সফল হয়েছে এবং সেই প্যাটার্ন মোতাবেক নতুন ডাটা সেটের জন্য আউটপুট বলে দিতে পারছে।

Medium কমিউনিটির ব্লগার @miloharper এর gist থেকে ফর্ক করা প্রোগ্রামটি নিচে দেয়া হলঃ  

```python

from numpy import exp, array, random, dot
 
class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
 
        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
 
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
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            # print("\nOutput of the Above Function After Sigmoid Applied: \n",output)
 
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output
            # print("\nTraining Set Output Matrix: \n", training_set_outputs)
            # print("\nError: Training Set Output Matrix 4x1 - Above Matrix 4x1 \n", error)
 
            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # print("\nAdjustment Matrix: \n", adjustment)
 
            # Adjust the weights.
            self.synaptic_weights += adjustment
 
    # The neural network thinks.
    def think(self, inputs):
        dot_product = dot(inputs, self.synaptic_weights)
        # print("\nDot Product of Input Matrix and Weight Matrix: \n",dot_product)
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot_product)
 
if __name__ == "__main__":
 
    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()
 
    print ("\n\nRandom starting synaptic weights: ")
    print (neural_network.synaptic_weights)
 
    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
 
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
 
    print ("\nNew synaptic weights after training: ")
    print (neural_network.synaptic_weights)
 
    # Test the neural network with a new situation.
    print ("\nConsidering new situation [1, 0, 0] -&amp;amp;amp;amp;amp;amp;amp;amp;amp;gt; ?: ")
    print (neural_network.think(array([1, 0, 0])))

```  

যদি প্রোগ্রামটির `if __name__ == “__main__”:` থেকে দেখা শুরু করেন তাহলে ধাপে ধাপে বুঝতে পারার কথা কিভাবে কোডের মাধ্যমে এই নিউরাল নেটওয়ার্ক তৈরি করা হয়েছে। আমি যথা সম্ভব আরেক্টূ সহজে ব্যাখ্যা করার চেষ্টা করছি। প্রোগ্রামের শুরুতেই numpy লাইব্রেরী যুক্ত  করা হয়েছে যাতে করে খুব সহজে ম্যাট্রিক্স পদ্ধতিতে কিছু ক্যালকুলেশনের কাজ করা যায় কারণ ন্যাটিভ পাইথনে ম্যাট্রিক্স টাইপের কোন ডাটা স্ট্রাকচার নাই। অন্যদিকে নিউরাল নেটওয়ার্কের গঠন মোতাবেক ইনপুট এবং ওয়েট নিয়ে গুন/যোগ ইত্যাদি করার সময় ম্যাট্রিক্স স্টাইল ভালো উপায়।

যেমন, এর মাধ্যমে আমাদের ট্রেনিং ডাটাসেট গুলোকে খুব সহজে ম্যাট্রিক্স এর রূপ দিতে পারি নিচের মত করে।  

```python
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T

```   

উল্লেখ্য – আউপুট কলামের ডাটা গুলোকে প্রথমে একটি 1×4 সাইজের ম্যাট্রিক্সে স্টোর করে তারপর ট্রান্সপোজ করে 4×1 সাইজে কনভার্ট করা হয়েছে যাতে ভিজুয়াল রিপ্রেজেন্টেশন মনে করা যেতে পারে এমন –

ইনপুট ম্যাট্রিক্স –

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-41-10-pm.png?w=720 "Ann") 

আউটপুট ম্যাট্রিক্স –

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-46-01-pm.png?w=720 "Ann")  

এভাবে ডাটা স্টোর করার ফলে আমরা একবারে পুরো ইনপুট ডাটা টেবিলকে আমাদের নিউরাল নেটওয়ার্কে ইনপুট দিয়ে খুব সহজে প্রত্যেকটা ইনপুট সেট (এ ক্ষেত্রে এক একটা row) -এর সাথে ওয়েট সেট ডট গুন করে একবারেই একটা আউটপুট ম্যাট্রিক্স পেয়ে যেতে পারি যেখানে ৪টা ইনপুট সেটের (৪টা row) জন্যই ৪টা আউটপুট ভ্যালু থাকবে 4×1 সাইজে। এতে করে প্রত্যেকটা Epoch এ পুরো অপারেশনটা একবার পুরোপুরি শেষ হবে।  এছাড়াও এই লাইব্রেরী থেকে আরও কিছু ফাংশনের সাহায্য নিয়ে কিছু অপারেশনকে সহজ বোধ্য করা হয়েছে।

স্ক্রিপ্ট হিসেবে এই প্রোগ্রামকে রান করালে ৫৮ নাম্বার লাইনে থেকেই এই প্রোগ্রামটির কার্যক্রম শুরু হয়। শুরুতেই NeuralNetwork ক্লাসের একটি অবজেক্ট তৈরি করা হয়েছে যার মাধ্যমে ফ্রেশ একটি নিউরাল নেটওয়ার্ক তৈরি করা যায়। তো, দেখে আসি  সেই ক্লাসের চেহারাটা। ৪নাম্বার লাইনে ক্লাসকে ডিফাইন করা হয়েছে। এর কন্সট্রাক্টরের মধ্যেই আমাদের সেই বহুল আলোচিত র‍্যান্ডম ওয়েট তিনটি তৈরি করা হচ্ছে।

যেহেতু আমাদের নিউরনের ৩টি ইনপুট তাই তিনটি ইনপুটের জন্য তিনটি ওয়েট নির্ধারণ করে ইনপুট গুলোর সাথে গুন করতে কাজ করার সুবিধার্থে 3×1 সাইজের একটি ম্যাট্রিক্স নেয়া/তৈরি করা হয়েছে synaptic_weights নামে। প্রথমবার অর্থাৎ ওয়েট অ্যাডজাস্ট হবার আগে এর চেহারা হতে পারে এমন –  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-54-27-pm.png?w=720 "Ann")  

*নোটঃ আপনি প্রোগ্রাম রান করানোর সময় আলাদা ভ্যালু পেতে পারেন কারন র‍্যান্ডমলি জেনারেটেড।*

এই ক্লাসের মধ্যে আরেকটি মেথড বানানো হয়েছে যার মাধ্যমে Sigmoid Function ব্যবহার করে ভ্যালু নরমালাইজেশন অর্থাৎ আউটপুট ভ্যালুকে 1 ও 0 মাঝে রাখা হয়। তার নিচেই আছে আরেকটি ফাংশন যার মাধ্যমে আমরা যেকোনো একটি আউটপুট ভ্যালুর জন্য Sigmoid Curve  এর Gradient বের করতে পারি। এটি কাজে লাগে ওয়েট অ্যাডজাস্টমেন্ট এর মান ঠিক করতে। উপড়ে এটা নিয়ে আলোচনা করা হয়েছে। এরপরেই আছে গুরুত্বপূর্ণ train ফাংশন যার মাধ্যমে আমাদের নিউরাল নেটওয়ার্কটি প্যাটার্ন চেনা শিখে নেয়।

প্রথমেই একটি লুপ চালানো হয়েছে যার মাণ নির্ধারণ করবে আপনি যতগুলো Epoch বা ট্রেনিং সাইকেল করাতে চান তার উপর। এখানে ১০০০০ বার Forward এবং Back Propogaion করাতে বলা হচ্ছে। ১০০০০ লুপের প্রথম iteration -এ  লুপের মধ্যের প্রথম কাজ হচ্ছে think ফাংশনের ব্যবহার করে এবং র‍্যান্ডম ওয়েটের উপর ভিত্তি করে একটা আউটপুট ম্যাট্রিক্স তৈরি করা যার মধ্যে নিউরনের হিসাব মোতাবেক পাওয়া আউটপুট গুলো থাকবে। এটির ডাইমেনশন হবে 4×1 অর্থাৎ ৪সেট ইনপুট ডাটার (৪টি row) জন্য ৪টি আউটপুট তথা নিচের মত একটি ম্যাট্রিক্স।  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-02-08-pm.png?w=720 "Ann") 

যদি think ফাংশনের কোড দেখি তাহলে দেখতে পারবো যে এখানে 4×3 সাইজের পুরো ইনপুট ডাটা টেবিল যাকে ম্যাট্রিক্সে কনভার্ট করা হয়েছে, তার সাথে 3×1 সাইজের ওয়েট ম্যাট্রিক্সের গুন করা হয়েছে। এতে করে বস্তুত প্রত্যেকটি ইনপুট সেট যেমন প্রথমত 0 0 1 এর সাথে তিনটি ওয়েট  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-54-27-pm.png?w=720 "Ann")  

– কে ডট গুন করা হয়েছে। আবার দ্বিতীয় ইনপুট সেট 1 1 1 এর সাথে একই ওয়েট ম্যাট্রিক্স  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-54-27-pm.png?w=720 "Ann")  

– কে ডট গুন করা হয়েছে। অর্থাৎ এভাবে সব গুলো ইনপুট কম্বিনেশনের সাথেই একবার করে ওই তিনটি ওয়েট ডট গুন করা হয়েছে। এভাবে যে আউটপুট ম্যাট্রিক্স পাওয়া যায় সেটাও কিন্তু 4×1 সাইজের ম্যাট্রিক্স। সেই ম্যাট্রিক্সকে একবার করে __sigmoid মধ্যে চালিয়ে নিয়ে ভ্যালু গুলোকে নরমালাইজ করা হয়েছে। তো, সব গুলো ইনপুট কম্বিনেশন এর সাথে ওয়েট গুলোর ডট গুন (গুন ও গুন গুলোর যোগ) করে নরমালাইজ করার পর নিচের মত একটি ম্যাট্রিক্স পাওয়া যাবে,  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-05-48-pm.png?w=720 "Ann")  

এই ম্যাট্রিক্সকে output ভ্যারিয়েবলে স্টোর করা হচ্ছে। এরপর এরর হিসাবের জন্য আমরা 4×1 সাইজের ট্রেনিং আউটপুট ম্যাট্রিক্স তথা,  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-46-01-pm.png?w=720 "Ann")  

থেকে উপরের 4×1 সাইজের output ম্যাট্রিক্স বিয়োগ করে নিচের মত একটি ম্যাট্রিক্স পেতে পারি,  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-08-31-pm.png?w=720 "Ann")  

এরপর এই এরর ম্যাট্রিক্স কে সাথে নিয়ে ইনপুট ডাটা সেট ম্যাট্রিক্স এবং Sigmoid Derivative কে কাজে লাগিয়ে অ্যাডজাস্টমেন্ট এর পরিমাণ বের করা হচ্ছে। এই অ্যাডজাস্টমেন্ট ম্যাট্রিক্সটিও ওয়েট ম্যাট্রিক্স এর মত 3×1 সাইজের। আর তাই train ফাংশনের শেষ লাইনে মুল ওয়েট ম্যাট্রিক্স এর সাথে এই অ্যাডজাস্ট ম্যাট্রিক্স যোগ করে ওয়েট ম্যাট্রিক্সে পরিবর্তন করে নেয়া হচ্ছে।

NeuralNetwork ক্লাসের কোড বোঝার পর আবারও ফিরে আসি পাইথন প্রোগ্রামের রেগুলার এক্সিকিউশন স্টেজে। ক্লাস ইনিসিয়ালাইজ করার পর পর্যবেক্ষণের স্বার্থে প্রথমবার সেট হওয়া র‍্যান্ডম ওয়েট ম্যাট্রিক্সকে প্রিন্ট করে দেখা হচ্ছে ওয়েট গুলো কি কি –  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-8-54-27-pm.png?w=720 "Ann")  

এরপর আমাদের ডাটা টেবিল থেকে ইনপুট এবং আউটপুট গুলোকে গুছিয়ে 4×3 সাইজের ট্রেনিং সেট ইনপুট এবং 4×1 সাইজের ট্রেনিং সেট আউটপুট ম্যাট্রিক্স বানিয়ে নেয়া হচ্ছে। এরপরেই উপড়ে আলোচ্য NeuralNetwork ক্লাসের অবজেক্ট neural_network –র মেথড, train এর মধ্যে এগুলো পাঠিয়ে দেয়া হচ্ছে। ১০০০০ বার চক্কর দেয়ার পর অপ্টিমাইজ ওয়েট ম্যাট্রিক্সটি কেমন রূপ ধারণ করলো সেটাও প্রিন্ট করা হচ্ছে।  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-13-10-pm.png?w=720 "Ann")  

সবশেষে, একটি নতুন ইনপুট সেট কে think ফাংশনে পাঠিয়ে আমাদের নিউরাল নেটওয়ার্ক এর কাছে আউটপুট জানতে চাওয়া হচ্ছে। এবার think ফাংশন, এই ইনপুট ডাটা সেট তথা 1×3 ম্যাট্রিক্সের সাথে আপডেটেড 3×1 ওয়েট ম্যাট্রিক্স এর ডট গুন করে Sigmoid অ্যাপ্লাই করে নরমালাইজ ডাটা তথা 1 থেকে 0 মধ্যের একটা ভ্যালুকে প্রিন্ট করে 1×1 সাইজের ম্যাট্রিক্স আকারে যেটা কিনা আমাদের নিউরাল নেটওয়ার্কের প্রেডিকশন।

আর সেটি হচ্ছে,  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-15-01-pm.png?w=720 "Ann")  

অর্থাৎ আমাদের নিউরাল নেটওয়ার্ক ভালোমতই ইনপুট ডাটা থেকে প্যাটার্ন খুঁজে তার উপর ভিত্তি করে পরবর্তী নতুন ইনপুট ডাটার জন্য তার আউটপুট কি হবে সেটা বলে দিতে পারছে।

আপনি যদি প্রথম iteration এর সব গুলো কাজের ধাপকে লগ করে দেখতে চান যে একটা ট্রেনিং লুপে কি কি ঘটছে তাহলে ৭৩ নাম্বার লাইনে 10000 এর পরিবর্তে 1 পাঠিয়ে এবং পুরো প্রোগ্রামের মধ্যে থাকা কমেন্ট করা প্রিন্ট স্টেটমেন্ট গুলোকে অ্যাক্টিভ করে দেখতে পারেন নিচের মত আউটপুট এবং সেগুলো ম্যানুয়ালি বিচার করতে পারেন।  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-22-26-pm.png "Ann")  

পুরো ১০০০০ বার লুপের পর অর্থাৎ ট্রেনিং শেষের পর ওয়েটেড ম্যাট্রিক্স এর ফাইনাল রূপ আসবে নিচের মত,  

![Ann](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-18-at-9-59-56-pm.png?w=720 "Ann")  