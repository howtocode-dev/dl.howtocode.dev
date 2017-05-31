## ডাটা ডাইমেনশন  
ডাটা ডাইমেনশন সম্পর্কে স্বচ্ছ ধারনা রাখতে হবে মাথায়। কোন ম্যাট্রিক্স বা টেনসরের ডাইমেনশন এর প্রসঙ্গ আসা মাত্রই যাতে কল্পনায় স্পষ্ট একটা ভিউ আসে ওই ডাটা অবজেক্টটার। তাহলে সব কিছু সহজ মনে হবে। যাই হোক, এরকম কিছু ডাইমেনশনকে আমরা কিছু ভ্যারিয়েবলে স্টোর করি এবার এবং সেলটি এক্সিকিউট করে নেই,

```python
# Cell 7
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10
```

img_size ভ্যারিয়েবলে আমাদের আলোচ্য ফটোগুলোর ডাইমেনশন স্টোর করছি। MNIST ডাটাসেটের ফটো গুলো 28x28 সাইজের ফটো। আসলে ইমেজের কন্টেক্সট থেকে বলতে, 28x28x1 সাইজের অর্থাৎ ফটো গুলো সাদা কালো এবং এর কালার চ্যানেল একটাই। রঙ্গিন ফটো হলে এদের ডাইমেনশন হত 28x28x3। RGB তিনটা রঙের তিনটা চ্যানেল এবং প্রত্যেক চ্যানেলের জন্য 28x28 সাইজের একগাদা পিক্সেল ভ্যালু। যা হোক, দ্বিতীয় ভ্যারিয়েবলে আমরা প্রত্যেকটি ইমেজের ফ্ল্যাট রিপ্রেজেন্টশন স্টোর করছি অর্থাৎ 28x28 সাইজের একটি ফটোর সবগুলো পিক্সেলকে যদি স্টোর করতে চাই তাহলে আমাদের img_size * img_size সাইজের একটি ওয়ান ডাইমেনশনাল অ্যারে বা ভেক্টর লাগবে। একটি টাপলে ইমেজের সেইপকে স্টোর করছি। আর শেষের ভ্যারিয়েবলে স্টোর করছি আমাদের যতগুলো আউটপুট ক্লাস দরকার সেই সংখ্যাটা। আমাদের ১০ ধরনের ক্লাসিফিকেশন দরকার, কারন ১০টাই ডিজিট দুনিয়াতে।

এ অবস্থায় একটু খুত খুতে লাগতে পারে এটা ভেবে যে - এইযে ফটো গুলো ইম্পরট করলাম এবং সেগুলোর উপর নাকি আবার কাজ করবো। সেগুলো আসলে দেখতে কেমন? ক্লিয়ার ভিউ তো লাগবে নাকি? ;)

নিচের কোড ব্লকটি পুরো একটি হেল্পার ফাংশন যার মাধ্যমে আমরা MNIST ডাটাসেটের ইমেজ গুলোকে রেন্ডার বা ডিসপ্লে করতে পারবো যেকোনো সময়। এখানে একটি 3x3 গ্রিডে মোট ৯টি ফটো এবং সেগুলোর সঠিক লেবেল ডিসপ্লে করানোর ব্যবস্থা করা হয়েছে।

```python
# Cell 8
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
```

তো, উপরের এই ফাংশনকে কাজে লাগিয়ে আমরা কিছু ফটো এবং সেগুলোর সাপেক্ষে সঠিক লেবেল রেন্ডার করে দেখি,

```python
# Cell 9
# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)
```

এই অবস্থায় নোটবুকের সেলটি এক্সিকিউট করলে নিচের মত আউটপুট আসবে,

<img class="aligncenter size-full wp-image-1769" src="https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-30-at-1-23-35-am.png" alt="" width="445" height="346" />
