# TensorFlow রান

অনেক হয়েছে গ্রাফ সাজানো। এবার পুরো গ্রাফকে রান করার পালা আর এর ম্যাজিক দেখার পালা। তাহলে সেশন তৈরি করে ফেলিঃ

```python
# Cell 23
session = tf.Session()
```

ভ্যারিয়েবল গুলোকে ইনিসিয়ালিয়াজ করে ফেলি,

```python
# Cell 24
session.run(tf.global_variables_initializer())
```

**কম রিসোর্সে অপটিমাইজেশন ফাংশনঃ**  
আমার জেনেছি যে - আমাদের ডাটাসেটে ৫০,০০০ ট্রেনিং ইমেজ আছে। যদি এই পুরো ডাটাসেটকে একবারেই আমাদের অপ্টিমাইজার ফাংশনের উপর দিয়ে দেই তাহলে কম্পিউটেশনে প্রচুর সময় লাগবে \(যদি না GPU বা হাই পাওয়ার CPU বা মেশিন হয়\) তাই আমরা যেটা করতে পারি, প্রত্যেকবার কিছু কিছু ইমেজ নিয়ে এর মধ্যে দিতে পারি। যেমন একবারে ১০০ করে দিতে পারি। এখানে ১০০ কে বলা হয় একটা batch.

```python
# Cell 25
batch_size = 100
```

এই কাজ করার জন্য আমরা একটা হেল্পার ফাংশন তৈরি করে নেই নিচে,

```python
# Cell 26
def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images. 
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph. # Note that the placeholder for y_true_cls is not set # because it is not used during training. 
        feed_dict_train = {x: x_batch,
                                            y_true: y_true_batch}
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer. 
        session.run(optimizer, feed_dict=feed_dict_train)
```

ভয়ের কি আছে? ফাংশনটা আসলে ৪ লাইনের :\) শুধু খেয়াল করার বিষয় হচ্ছে feed\_dict\_train ভ্যারিয়েবলটা। এটার মাধ্যমেই কিন্তু প্লেসহোল্ডারের মধ্যে সত্যিকারের ইনপুট দেয়া হচ্ছে। এক্ষেত্রে x আর y\_true কিন্তু আমরাই ডিক্লেয়ার করেছিলাম প্লেসহোল্ডার হিসেবে। আরেকটা হেল্পার ফাংশন আমারা বানিয়ে নিতে পারি পারফর্মেন্স শো করার জন্য. এটার ক্ষেত্রে ফিড ডিকশনারি হবে টেস্ট ইমেজ গুলো নিয়ে। নিচের মত,

```python
# Cell 27
feed_dict_test = {x: data.test.images, y_true: data.test.labels,
                            y_true_cls: data.test.cls}
```

আর ফাংশনটি হবে,

```python
# Cell 28
def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))
```

**ওয়েট ভিজুয়ালাইজ করাঃ**  
এই পার্টটি অপশনাল। যদি কেউ দেখতে চান ওয়েট ম্যাট্রিক্স গুলো দেখতে কেমন হচ্ছে তাহলে নিচের হেল্পার ফাংশন ব্যবহার করা যেতে পারে। ওয়েট ম্যাট্রিক্স যেহেতু একটা ম্যাট্রিক্স আর আমি ম্যাট্রিক্স মানেই মনে করি ইমেজ তাই এটাও দেখতে ইমেজের মত হবে। হোক না দুই কালার ওয়ালা।

```python
# Cell 29
def plot_weights():
    # Get the values for the weights from the TensorFlow variable. w = session.run(weights)
    # Get the lowest and highest values for the weights. # This is used to correct the colour intensity across # the images so they can be compared with each other. w_min = np.min(w)
    w_max = np.max(w)
    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused. fig, axes = plt.subplots(3, 4) fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots. 
        if i<10:
            # Get the weights for the i'th digit and reshape it. # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)
                                # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i)) 
            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
```

> পরের টিউটোরিয়ালে কনভলিউশন বোঝার জন্য ওয়েট ম্যাট্রিক্স এর ভিজুয়ালাইজেশন বোঝা উপকারী।

**কোন রকম অপ্টিমাজেশনের আগেই পারফর্মেন্স চেক করিঃ**  
আমরা যদি এই অবস্থায় নোটবুকের নতুন একটি সেলে নিচের কোড এক্সিকিউট করে accuracy দেখতে চাই,

```python
# Cell 30
print_accuracy()
```

তাহলে আউটপুট আসবে,

```python
Accuracy on test-set: 9.8%
```

কারন কি? মডেল ট্রেনিং করার আগেই কিভাবে শতকরা 10 ভাগ সঠিক উত্তর দেয়া শুরু করলো? আজব না? কারন হচ্ছে - আমাদের ওয়েট বায়াস শূন্য। তাই মডেল সব ইমেজকে প্রেডিক্ট করে শূন্য হিসেবে। কাকতালীয় ভাবে টেস্ট ইমেজ গুলোর মধ্যে শতকরা ১০ ভাগের মত ইমেজ ছিল শূন্যের। তাই সেগুলোর ক্ষেত্রে যখন প্রেডিক্টেড আর ট্রু ক্লাস মিলে গেছে, তাই accuracy আসতেছে 10% এর মত।

> ঝড়ে বক মরে ফকিরের কেরামতি বারে

**অপটিমাইজেশন শুরু করিঃ**  
আমি এই পোস্টের শুরুর দিকের একটা উদাহরনেও একটা লুপকে ১টা সাইকেলে আটকে রেখে ভ্যালু গুলো নিয়ে যাচাই বাছাই এর কথা বলেছিলাম। এবারও সেরকম একটা এক্সপেরিমেন্ট করা যায়। আমরা একটি মাত্র অপটিমাইজেশন ইটারেশন করবো শুরুতে।

```python
# Cell 31
optimize(num_iterations=1)
```

```python
# Cell 32
print_accuracy()
```

```python
Accuracy on test-set: 40.7%
```

একটা ইটারেশনেই প্রায় ৪০% সঠিক রেজাল্ট দিতে শিখেছে এই মডেল। বলে নেয়া ভালো - একটা ইটারেশনে কিন্তু এক ব্যাচ পরিমাণ ইমেজ নিয়ে কাজ করে মডেলটি। প্রত্যেক ইটারেশনে নতুন এবং পরবর্তী ব্যাচ \(১০০টি\) নিয়ে কাজ করে। optimize ফাংশনের কোড খেয়াল করুন। তো, এ অবস্থায় ওয়েট গুলো দেখতে চাইলে,

```python
# Cell 33
plot_weights()
```

আউটপুট আসবে নিচের মত,

![](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-30-at-1-30-14-am.png)

এখানে বলে নেয়া ভালো পজিটিভ ওয়েট গুলোকে লাল রং এবং নেগেটিভ ওয়েট গুলোকে নীল রং -এ প্রকাশ করা হয়েছে। মনে আছে অনেক আগের সেকশনে আমরা এরকম ওয়েট ম্যাট্রিক্স দিয়ে কনভলিউশনের ব্যাসিক ধারনা নিয়েছিলাম? যেখানে ম্যাট্রিক্স গুলো দেখতে ছিল \[\[+ -\], \[- +\]\] এরকম? এগুলোকেই ফিল্টার হিসাবে বলা হবে কনভলিউশন লেয়ারে। এখানে আমাদের মডেল ট্রেনিং করে শিখে এরকম ওয়েট ম্যাট্রিক্স ধরে নিয়েছে এবং দেখেছে যে এরকম ওয়েট ম্যাট্রিক্স হলে সেই রিলেটেড ফটো গুলোর সাথে রিয়েকশন পজিটিভ হয়। অর্থাৎ, যদি একটা 0 ওয়ালা ইমেজের সাথে এই ফিল্টারের দেখা হয় \(এক্ষেত্রে ডাইরেক্ট x\*W. কোন নির্দিষ্ট পার্টের সাথে কনভলিউশন নয়\) তাহলে এই ফিল্টার সেই ফটোর সাথেই পজিটিভ রিয়েকশন করবে যার মধ্যে একটি সার্কেল টাইপ দাগ আছে। আর সেগুলোর সাথে নেগেটিভ রিয়েকশন করবে যেগুলোর মাঝখানটায় এক গাদা কালি আছে। তার মানে সে শূন্য লেখা আছে এমন ফটোর সাথে বেশি পজিটিভ ভ্যালু তৈরি করবে। এসব আরও বিস্তারিত বোঝা যাবে যাবে পরের টিউটোরিয়ালে যেখানে কনভলিউশনাল নিউরাল নেটওয়ার্ক ডিজাইন করা হবে এই মডেলকেই আরও ইফিসিয়েন্ট করার জন্য।

যা হোক, এবার ১০০০ অপটিমাইজেশন ইটারেশন করে দেখা যাকঃ

```python
# Cell 34
# We have already performed 1 iteration already. 
optimize(num_iterations=999)
```

```python
# Cell 35
print_accuracy()
```

```python
Accuracy on test-set: 91.7%
```

খেয়াল করুন, শুধুমাত্র লিনিয়ার মডেল ডিজাইন করেও ৯১% Accuracy পাওয়া গেছে। এটা সম্ভব হয়েছে ডিপ লার্নিং এর কারনেই। এখানে আমরা ইমেজ থেকে ফিচার এক্সট্র্যাক্ট করে দেই নি। শুধু ডাইরেক্ট পিক্সেল ভ্যালু গুলোকে ইনপুট লেয়ারে দিয়ে আউটপুট লেয়ারে ট্রু ক্লাস দিয়ে ট্রেনিং করে মডেল ভ্যারিয়েবল গুলোকে অ্যাডজাস্ট করতে বলেছি। এতেই সে ওয়েট ম্যাট্রিক্স ধারনা করা শিখে গেছে। চাইলে এই অবস্থাতেও ওয়েট গুলো ভিজুয়ালাইজ করে দেখতে পারেন আর কোন প্রশ্ন থাকলে করতে পারেন।

```python
# Cell 36
plot_weights()
```

![](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-30-at-1-31-16-am.png)

এখানে দেখা যাচ্ছে ওয়েট ম্যাট্রিক্স গুলো আরও একটু জটিল হিসাবে মগ্ন। অর্থাৎ এমন না যে সার্কেল ধরে পজিটিভ ওয়েট সেট করেছে \(যেমন 0 এর ক্ষেত্রে\)। বরং একটু ছাড়া ছাড়া ভাবে। এটা সে করতে বাধ্য হয়েছে এক এক জনের এক এক রকম শূন্য লেখার সঙ্গে নিজেকে মানিয়ে নিতে গিয়ে।

অনেক হল গবেষণা। তো, এবার আমরা TensorFlow এর সেশন ক্লোজ করতে পারি নিচের মত।

```python
# Cell 37
session.close()
```

> উপরের আলোচ্য ধাপ গুলো নিয়ে পূর্ণ .ipynb ডকুমেন্টটি পাওয়া যাবে [এখানে](https://github.com/nuhil/deep-learning-research/blob/master/TF-Linear-MNIST.ipynb)

**পূর্ণ প্রোগ্রামঃ** যারা নোটবুকে ধাপে ধাপে এই কোড ব্লক গুলো এক্সিকিউট করেছেন বোঝার জন্য এবং এখন গোছানো একটা প্রোগ্রাম চান যেকোনো জায়গায় রান করার জন্য - [ক্লিক করুন এখানে](https://github.com/nuhil/deep-learning-research/blob/master/tf-mnist-lm-digit-classi.py)।

