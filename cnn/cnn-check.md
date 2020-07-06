# পরীক্ষা করে দেখা

আবার Xএর ফটো ইনপুট হিসেবে দিয়েই পরীক্ষা করি পুরো কনভলিউশনাল নিউরাল নেটওয়ার্ক আসলেই Xকে চিনতে পারে কিনা।&lt;/p&gt;

![](https://nuhil.files.wordpress.com/2017/05/screen-shot-2017-05-20-at-9-15-54-pm.png?w=687)

খেয়াল করুন, প্রথমে X এর ইমেজের উপর আমাদের \ / ফিল্টার চালিয়ে দেখা হয়েছে এবং পুলিং লেয়ারের কাছে শুধু ৪টি অবস্থা ভ্যালিড বা ফায়ার হয়েছে। ব্যাক স্ল্যাস ফিল্টার দিয়ে ঘোরার সময় যে দুটা অবস্থায় ব্যাক স্ল্যাস পাওয়া গেছে এবং ফরওয়ার্ড স্ল্যাস ফিল্টার দিয়ে ট্রাভেল করার সময় যে দুটা জায়গায় ফরওয়ার্ড স্ল্যাসের অস্তিত্ব পাওয়া গেছে \(তাই পুলিং লেয়ারের ইনপুটে ৪টি কেইস\)। এরপর পুলিং লেয়ার 2x2 পিক্সেলেটেড ইমেজ কে বিশেষ ম্যাট্রিক্সে কনভার্ট করছে এবং পরীক্ষার কোন এক সিচুয়েশনে এই ম্যাট্রিক্সটি ডান পাশের ফুলি কানেক্টেড লেয়ারের প্রত্যেকটি ফিল্টারের সাথে মাল্টিপ্লাইড হচ্ছে। ফাইনালি সে ডান পাশে অর্থাৎ আউপুটপুট লেয়ারে রেজাল্ট হিসেবে জানাচ্ছে তার পাওয়া স্কোর গুলো। আর স্কোর দেখে খুব সহজেই বুঝে নেয়া যাচ্ছে X এর উপরেই এই CNN এর কনফিডেন্স বেশি :\)

বার বার মনে করিয়ে দিচ্ছি, এখানে বেশ কিছু হেল্পার ফাংশনের কাজ এড়িয়ে যাওয়া হয়েছে শুধু নিউরাল নেটওয়ার্কের ওয়ার্কিং প্রিন্সিপল সহজে বোঝানোর জন্য। যেমন - কম্পিউটারকে পারফেক্ট ফিল্টার বুঝতে, কনভলিউশন করাতে, ফুলি কানেক্টেড লেয়ারের ওয়েট/এইজ উদ্ধার করতে লক্ষ্য লক্ষ্য বার ঘুরে ফিরে কাজ করতে হয়। কারণ, শুরুতেই কম্পিউটার সব কিছুর জন্য \(ফিল্টার, কনভলিউশন ইত্যাদি\) র‍্যান্ডম কিছু ভ্যালু ধরে নেয়। তারপর ট্রেনিং ডাটা সেট এ যেহেতু প্রশ্ন উত্তর দুটাই আছে, তাই সেখান থেকে এরর কন্সিডার করে করে এবং সেই অনুযায়ী সব ভ্যালু অ্যাডজাস্ট করে করে ফাইনালি এরকম স্ট্যাবল একটা স্টেজে আসে। এরর এর উপর ভিত্তি করে ভ্যালু অ্যাডজাস্ট করা নির্ভর করে Gradient Descent এরউপর। এ সম্পর্কে বাংলায় পড়তে চাইলে [এখানে ক্লিক করুন](https://ml.howtocode.com.bd/linear_regression/linear_regression_2.html)।

**বাস্তব জগতে CNN** অনেক তো খেলনা জগতের সমস্যা উদ্ধার করলাম আমরা। আসলেই রিয়েল লাইফ সিচুয়েশনে কিভাবে CNN কাজ করে তার একটা ধারনা নেই এখন। কারন, বাস্তবে কম্পিউটারে লক্ষ্য লক্ষ্য পিক্সেল যেমন আছে তেমনি সব ফটো আমাদের আরাম দেয়ার জন্য 3x3 পিক্সেল নিয়ে বসে নাই। কয়েক মেগা পিক্সেলের ইমেজ এখন সবার কাছেই। তাই আমাদের যেমন নিউরাল নেটওয়ার্ক ডিজাইন করতে হবে খুব বুদ্ধি করে, তেমনি কম্পিউটারকেও রেডি থাকতে হবে বিনা ইস্যুতে কোটি কোটি বার একই বোরিং হিশাব করতে আর লুপের উপর থাকতে।

![](https://nuhil.files.wordpress.com/2017/05/screen-shot-2015-11-07-at-7-26-20-am.png?w=687)

উপরে একটা পূর্ণ CNN এর ব্লক ডায়াগ্রাম দেখানো হয়েছে। প্রথমেই বাম পাশে একটি নৌকার ছবি ইনপুট দেয়া হচ্ছে এবং এই নেটওয়ার্কে দুই স্টেজে Convolution এবং Pooling এর কাজ করা হয়েছে \(প্রয়োজনে আরও হতে পারে\)। তো, প্রথম কনভলিউশন এবং পুলিং এর সময় এই ফটো থেকে কিছু গুরুত্বপূর্ণ পিক্সেল গুচ্ছ বা অবজেক্টের অংশ বিশেষ আলাদা করে নিয়ে নেয়া সম্ভব হয়। আবারো, কনভলিউশন এবং পুলিং লেয়ারের সাহায্যে যতটা সম্ভব সিমপ্লিফ্যায়েড কিন্তু অর্থবহ ইমেজে নিয়ে আসা হয়। এরপর সেই লেয়ারের আউটপুট কে ফুলি কানেক্টেড এক বা একাধিক লেয়ারে ইনপুট হিসেবে দিয়ে সবার সাথে সবার গুন/যোগ করে স্কোর জেনারেট করা হয়। ভ্রমণটা ট্রেনিং টাইপের হলে স্কোর এবং আসল আউটপুট এর পার্থক্য দেখে চক্কর দিতে থাকে এরর কমানোর জন্য। আর ভ্রমণটা ট্রেনিং শেষে প্রেডিকশনের জন্য হলে, একটা স্কোর দিয়ে দেয় যার মাধ্যমে আমরা চিনতে পারি যে ফটোটা নৌকার।
